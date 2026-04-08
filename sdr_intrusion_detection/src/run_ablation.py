#!/usr/bin/env python3
"""
run_ablation.py — Complete ablation study runner.

Generates data, trains all models, benchmarks inference, and produces all figures.

Usage:
    python -m src.run_ablation                              # Full pipeline
    python -m src.run_ablation --skip_data                  # Skip data generation
    python -m src.run_ablation --quick --duration 2 --epochs 10  # Quick smoke test
"""

import argparse
import json
import os
import time
from datetime import timedelta

try:
    import optuna
except ImportError:
    optuna = None

from src.sim_system import generate_dataset
from src.train import train_all_models, train_model
from src.benchmark_edge import benchmark_all_models
from src.evaluate import generate_all_figures


def merge_benchmark_into_ablation(results_dir: str, benchmark_results: dict) -> None:
    """Attach inference, size, and parameter metrics into ablation_results.json."""
    ablation_path = os.path.join(results_dir, "ablation_results.json")
    if not os.path.exists(ablation_path):
        return

    with open(ablation_path, 'r') as f:
        ablation = json.load(f)

    for model_name, metrics in benchmark_results.items():
        if model_name in ablation and isinstance(ablation[model_name], dict):
            ablation[model_name]['benchmark'] = metrics

    with open(ablation_path, 'w') as f:
        json.dump(ablation, f, indent=2, default=str)


def run_hyperparameter_tuning(
    data_dir: str,
    results_dir: str,
    n_trials: int = 20,
    epochs: int = 15,
) -> dict:
    """Run Optuna study on the target SOTA model."""
    if optuna is None:
        print("\n  ❌ Optuna is not installed! Run: pip install optuna")
        print("  ⏩ Falling back to default hyperparameters.\n")
        return {}

    print(f"\n{'='*60}")
    print(f"  PHASE 1.5: Hyperparameter Tuning (Optuna)")
    print(f"  Model: dual_branch_fusion | Trials: {n_trials} | Base Epochs: {epochs}")
    print(f"{'='*60}")

    def objective(trial):
        # Define search space
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
        label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.15)
        branch_dim = trial.suggest_categorical('branch_dim', [48, 64, 80])
        spec_branch_dim = trial.suggest_categorical('spec_branch_dim', [64, 80, 96])
        fusion_dim = trial.suggest_categorical('fusion_dim', [48, 64, 80])
        dropout = trial.suggest_float('dropout', 0.05, 0.25)
        
        cfg = {
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'weight_decay': weight_decay,
            'label_smoothing': label_smoothing,
            'patience': max(3, epochs // 3),
            'n_folds': 2,  # Quick 2-fold for speed
            'model_kwargs': {
                'branch_dim': branch_dim,
                'spec_branch_dim': spec_branch_dim,
                'fusion_dim': fusion_dim,
                'dropout': dropout,
            },
            # Use minimal logging
        }
        
        res = train_model(
            model_name="dual_branch_fusion", 
            data_dir=data_dir, 
            output_dir=os.path.join(results_dir, "optuna_temp"),
            config=cfg
        )
        return res.get('best_accuracy', 0.0)

    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(direction='maximize', study_name="RF_CNN_Tuning")
    study.optimize(objective, n_trials=n_trials)
    
    print(f"\n  🎯 BEST HIGHEST ACCURACY: {study.best_value:.4f}")
    print("  🏆 WINNING PARAMS:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
        
    best_params = dict(study.best_params)
    model_kwargs = {
        key: best_params.pop(key)
        for key in ['branch_dim', 'spec_branch_dim', 'fusion_dim', 'dropout']
        if key in best_params
    }
    if model_kwargs:
        best_params['model_kwargs'] = model_kwargs

    return best_params


def run_full_pipeline(
    data_dir: str = "dataset/simulated",
    results_dir: str = "results/models",
    figures_dir: str = "results/figures",
    duration: float = 10.0,
    snr_db: float = 20.0,
    epochs: int = 50,
    batch_size: int = 128,
    skip_data: bool = False,
    skip_train: bool = False,
    skip_benchmark: bool = False,
    quick: bool = False,
    tune: bool = False,
    n_trials: int = 20,
    n_folds: int = 5,
    extended_baselines: bool = False,
):
    """
    Run the complete pipeline: generate → tune → train → benchmark → evaluate.
    """
    start_time = time.time()
    all_results = None

    if quick:
        duration = min(duration, 2.0)
        epochs = min(epochs, 10)
        print("\n⚡ QUICK MODE: Reduced duration & epochs for testing\n")

    # =========================================
    # Phase 1: Generate Data
    # =========================================
    if not skip_data:
        print("\n" + "=" * 60)
        print("  PHASE 1: Generating Simulated Dataset")
        print("=" * 60)

        generate_dataset(
            output_dir=data_dir,
            duration_sec=duration,
            snr_db=snr_db,
        )
    else:
        print("\n  ⏩ Skipping data generation (--skip_data)")

    # =========================================
    # Phase 1.5: Hyperparameter Tuning
    # =========================================
    tuned_params = {}
    if tune:
        tuned_params = run_hyperparameter_tuning(
            data_dir=data_dir, 
            results_dir=results_dir, 
            n_trials=n_trials if not quick else 2,
            epochs=min(epochs, 15)  # Cap epochs for tuning iteration speed
        )
    else:
        # User didn't request tuning, proceed normally
        pass

    # =========================================
    # Phase 2: Train All Models
    # =========================================
    if not skip_train:
        print("\n" + "=" * 60)
        print("  PHASE 2: Training All Models (Ablation Study)")
        print("=" * 60)

        config = {
            'epochs': epochs,
            'batch_size': batch_size,
            'label_smoothing': 0.05,
            'patience': max(5, epochs // 5),
            'n_folds': n_folds,
            'model_kwargs': {
                'branch_dim': 64,
                'spec_branch_dim': 80,
                'fusion_dim': 64,
                'dropout': 0.15,
            },
        }
        
        # Inject Optuna parameters if we tuned them!
        if tuned_params:
            print(f"  Injecting Optuna hyperparameters: {tuned_params}")
            config.update(tuned_params)
            
            # Reset n_folds to 5 for the final ablation study (Optuna did 2)
            if 'n_folds' in config:
                del config['n_folds']

        all_results = train_all_models(
            data_dir,
            results_dir,
            config,
            include_extended_baselines=extended_baselines,
        )
    else:
        print("\n  ⏩ Skipping training (--skip_train)")

    # =========================================
    # Phase 3: Benchmark Inference
    # =========================================
    if not skip_benchmark:
        print("\n" + "=" * 60)
        print("  PHASE 3: Edge Inference Benchmarking")
        print("=" * 60)

        benchmark_model_names = list(all_results.keys()) if all_results else None
        benchmark_results = benchmark_all_models(
            results_dir,
            model_names=benchmark_model_names,
            data_dir=data_dir,
        )
        merge_benchmark_into_ablation(results_dir, benchmark_results)
    else:
        print("\n  ⏩ Skipping benchmark (--skip_benchmark)")

    # =========================================
    # Phase 4: Generate Figures
    # =========================================
    print("\n" + "=" * 60)
    print("  PHASE 4: Generating Paper Figures")
    print("=" * 60)

    generate_all_figures(results_dir, data_dir, figures_dir)

    # =========================================
    # Summary
    # =========================================
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  ✅ PIPELINE COMPLETE")
    print(f"  Total time: {timedelta(seconds=int(elapsed))}")
    print(f"  Data:       {data_dir}/")
    print(f"  Models:     {results_dir}/")
    print(f"  Figures:    {figures_dir}/")
    print(f"{'='*60}\n")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Run complete ablation study pipeline"
    )
    parser.add_argument("--data_dir", type=str, default="dataset/simulated")
    parser.add_argument("--results_dir", type=str, default="results/models")
    parser.add_argument("--figures_dir", type=str, default="results/figures")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Data generation duration per class (seconds)")
    parser.add_argument("--snr_db", type=float, default=20.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--skip_data", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_benchmark", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (2s data, 10 epochs)")
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter optimization prior to training")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of Optuna trials for hyperparameter tuning")
    parser.add_argument("--extended_baselines", action="store_true",
                        help="Include 10 additional torchvision spectrogram baselines")
    args = parser.parse_args()

    run_full_pipeline(**vars(args))


if __name__ == "__main__":
    main()

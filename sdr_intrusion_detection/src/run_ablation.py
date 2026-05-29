"""
run_ablation.py — Complete IEEE Ablation Study
===============================================
Phase 1: Backbone Comparison (19 architectures, fixed hyperparams)
Phase 2: Hyperparameter Sweep via Optuna (custom model only)
Phase 3: Cross-dataset generalization (if source splits exist)
"""
import argparse
import json
import os
import time

import optuna
from src.train import train_model
from src.evaluate import evaluate_model
from src.model import TIMM_MODEL_MAP

CUSTOM_MODEL = 'SDR_Custom_CoordASPP_Focal'
DEFAULT_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'unified_dataset'))


# ============================================================
# PHASE 1: Backbone Comparison
# ============================================================
def phase_backbone(data_dir, epochs, batch_size):
    """Train all 19 architectures under identical default hyperparams."""
    models = [CUSTOM_MODEL] + list(TIMM_MODEL_MAP.keys())

    print("=" * 80)
    print(f"  PHASE 1: BACKBONE COMPARISON ({len(models)} architectures)")
    print("=" * 80)

    summary = {}
    for model_name in models:
        print(f"\n>>> [{model_name}] Starting ...")
        args = argparse.Namespace(
            data_dir=data_dir, model=model_name,
            epochs=epochs, batch_size=batch_size,
            lr=1e-3, focal_gamma=2.0, optimizer='adamw',
            weight_decay=1e-4, dropout=0.3
        )
        try:
            log = train_model(args)
            summary[model_name] = {
                "best_val_acc": log["best_val_acc"],
                "profile": log["profile"]
            }
            evaluate_model(args)
            print(f"  [{model_name}] Done — Val Acc: {log['best_val_acc']:.2f}%")
        except Exception as e:
            print(f"  [{model_name}] FAILED: {e}")
        time.sleep(3)

    os.makedirs('results', exist_ok=True)
    with open('results/ablation_backbone.json', 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"\nPhase 1 complete. Results -> results/ablation_backbone.json")
    return summary


# ============================================================
# PHASE 2: Hyperparameter Sweep (Optuna)
# ============================================================
def _optuna_objective(trial, data_dir, epochs):
    """Single Optuna trial — tune our custom model."""
    lr         = trial.suggest_categorical('lr', [1e-4, 5e-4, 1e-3, 5e-3])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    optimizer  = trial.suggest_categorical('optimizer', ['adamw', 'sgd'])
    gamma      = trial.suggest_categorical('focal_gamma', [1.0, 2.0, 3.0])
    wd         = trial.suggest_categorical('weight_decay', [1e-5, 1e-4, 1e-3])
    dropout    = trial.suggest_categorical('dropout', [0.2, 0.3, 0.5])

    args = argparse.Namespace(
        data_dir=data_dir, model=CUSTOM_MODEL,
        epochs=epochs, batch_size=batch_size,
        lr=lr, focal_gamma=gamma, optimizer=optimizer,
        weight_decay=wd, dropout=dropout
    )

    log = train_model(args)
    return log["best_val_acc"]


def phase_hparam(data_dir, epochs, n_trials=20):
    """Run Optuna hyperparameter sweep on the custom model."""
    print("=" * 80)
    print(f"  PHASE 2: HYPERPARAMETER SWEEP ({n_trials} Optuna trials)")
    print("=" * 80)

    study = optuna.create_study(
        direction='maximize',
        study_name='SDR_CoordASPP_HPO',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(
        lambda trial: _optuna_objective(trial, data_dir, epochs),
        n_trials=n_trials,
        show_progress_bar=True
    )

    # Save results
    os.makedirs('results', exist_ok=True)
    trials = []
    for t in study.trials:
        trials.append({
            "number": t.number,
            "value": t.value,
            "params": t.params
        })

    with open('results/ablation_hparam.json', 'w') as f:
        json.dump({
            "best_trial": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "all_trials": trials
        }, f, indent=4)

    print(f"\nPhase 2 complete.")
    print(f"  Best Accuracy : {study.best_value:.2f}%")
    print(f"  Best Params   : {study.best_params}")
    print(f"  Results -> results/ablation_hparam.json")
    return study


# ============================================================
# PHASE 3: Cross-Dataset Generalization
# ============================================================
def phase_cross_dataset(epochs, batch_size):
    """Train on one sensor source, test on the other."""
    base = r'C:\Users\hkphi\OneDrive\Desktop\WORK\EndSemDSLabK'
    usrp_dir = os.path.join(base, 'unified_dataset')  # Primary unified
    radar_dir = os.path.join(base, '_radar_staging')    # Radar only

    pairs = []
    if os.path.isdir(usrp_dir):
        pairs.append(("Unified", usrp_dir))
    if os.path.isdir(radar_dir):
        pairs.append(("Radar", radar_dir))

    if len(pairs) < 2:
        print("  [SKIP] Cross-dataset requires at least 2 separate source directories.")
        print("         Found:", [p[0] for p in pairs])
        return {}

    print("=" * 80)
    print("  PHASE 3: CROSS-DATASET GENERALIZATION")
    print("=" * 80)

    results = {}
    for train_name, train_dir in pairs:
        for test_name, test_dir in pairs:
            if train_name == test_name:
                continue
            tag = f"Train_{train_name}_Test_{test_name}"
            print(f"\n>>> {tag}")
            args = argparse.Namespace(
                data_dir=train_dir, model=CUSTOM_MODEL,
                epochs=epochs, batch_size=batch_size,
                lr=1e-3, focal_gamma=2.0, optimizer='adamw',
                weight_decay=1e-4, dropout=0.3
            )
            try:
                log = train_model(args)
                # Evaluate on the OTHER dataset's test split
                eval_args = argparse.Namespace(
                    data_dir=test_dir, model=CUSTOM_MODEL,
                    batch_size=batch_size
                )
                evaluate_model(eval_args)
                results[tag] = log["best_val_acc"]
            except Exception as e:
                print(f"  {tag} FAILED: {e}")

    with open('results/ablation_cross_dataset.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nPhase 3 complete. Results -> results/ablation_cross_dataset.json")
    return results


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IEEE Ablation Study — 3 Phases")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "backbone", "hparam", "cross"],
                        help="Which ablation phase to run")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of Optuna trials for hparam phase")
    args = parser.parse_args()

    if args.phase in ("all", "backbone"):
        phase_backbone(args.data_dir, args.epochs, args.batch_size)
    if args.phase in ("all", "hparam"):
        phase_hparam(args.data_dir, args.epochs, args.n_trials)
    if args.phase in ("all", "cross"):
        phase_cross_dataset(args.epochs, args.batch_size)

    print("\n" + "=" * 80)
    print("  ALL ABLATION PHASES COMPLETE!")
    print("  Backbone results  -> results/ablation_backbone.json")
    print("  HPO results       -> results/ablation_hparam.json")
    print("  Cross-dataset     -> results/ablation_cross_dataset.json")
    print("  W&B dashboard     -> https://wandb.ai")
    print("=" * 80)

"""
run_ablation.py — Complete IEEE Ablation Study (Parallel-Capable)
================================================================
Phase 1: Backbone Comparison (19 architectures, parallel subprocess workers)
Phase 2: Hyperparameter Sweep via Optuna (custom model only)
Phase 3: Cross-dataset generalization (if source splits exist)

Parallelism:
  Phase 1 launches each model as an independent `python -m src.train`
  subprocess.  A configurable --max_workers (default 2) limits concurrency
  to avoid GPU OOM on a shared A100 32GB.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import optuna
from src.train import train_model
from src.evaluate import evaluate_model
from src.model import TIMM_MODEL_MAP

CUSTOM_MODEL = 'SDR_Custom_CoordASPP_Focal'
DEFAULT_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'unified_dataset'))

# Root of the cloned repository (sdr_intrusion_detection/)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


# ============================================================
# PHASE 1: Backbone Comparison  (PARALLEL)
# ============================================================
def _train_single_model_subprocess(model_name, data_dir, epochs, batch_size, lr):
    """
    Train ONE model in a fresh subprocess.
    Returns (model_name, result_dict) or (model_name, error_string).
    """
    cmd = [
        sys.executable, '-m', 'src.train',
        '--data_dir', data_dir,
        '--model', model_name,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--lr', str(lr),
    ]
    log_path = os.path.join(_PROJECT_ROOT, 'results', 'logs', f'subprocess_{model_name}.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    try:
        with open(log_path, 'w') as logf:
            proc = subprocess.run(
                cmd,
                cwd=_PROJECT_ROOT,
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=7200  # 2h safety
            )
        # After training, read the JSON history produced by train.py
        history_path = os.path.join(_PROJECT_ROOT, 'results', 'logs', f'history_{model_name}.json')
        if os.path.isfile(history_path):
            with open(history_path) as f:
                result = json.load(f)
            return (model_name, result)
        else:
            return (model_name, f"Process exited with code {proc.returncode} but no history JSON found. Check {log_path}")
    except subprocess.TimeoutExpired:
        return (model_name, f"TIMEOUT after 2 hours. Check {log_path}")
    except Exception as e:
        return (model_name, str(e))


def phase_backbone(data_dir, epochs, batch_size, max_workers=2):
    """Train all 19 architectures using parallel subprocesses."""
    models = [CUSTOM_MODEL] + list(TIMM_MODEL_MAP.keys())

    print("=" * 80)
    print(f"  PHASE 1: BACKBONE COMPARISON ({len(models)} architectures)")
    print(f"  Parallelism: max_workers = {max_workers}")
    print("=" * 80)

    summary = {}
    failed = []
    lr = 1e-3

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _train_single_model_subprocess,
                name, data_dir, epochs, batch_size, lr
            ): name
            for name in models
        }

        for future in as_completed(futures):
            model_name = futures[future]
            try:
                name, result = future.result()
                if isinstance(result, dict):
                    summary[name] = {
                        "best_val_acc": result.get("best_val_acc", 0),
                        "profile": result.get("profile", {})
                    }
                    print(f"  ✓ [{name}] Done — Val Acc: {result.get('best_val_acc', 0):.2f}%")
                else:
                    failed.append(name)
                    print(f"  ✗ [{name}] FAILED: {result}")
            except Exception as exc:
                failed.append(model_name)
                print(f"  ✗ [{model_name}] Exception: {exc}")

    os.makedirs(os.path.join(_PROJECT_ROOT, 'results'), exist_ok=True)
    out_path = os.path.join(_PROJECT_ROOT, 'results', 'ablation_backbone.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\nPhase 1 complete. {len(summary)} succeeded, {len(failed)} failed.")
    if failed:
        print(f"  Failed models: {failed}")
    print(f"  Results -> {out_path}")
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
    os.makedirs(os.path.join(_PROJECT_ROOT, 'results'), exist_ok=True)
    trials = []
    for t in study.trials:
        trials.append({
            "number": t.number,
            "value": t.value,
            "params": t.params
        })

    out_path = os.path.join(_PROJECT_ROOT, 'results', 'ablation_hparam.json')
    with open(out_path, 'w') as f:
        json.dump({
            "best_trial": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "all_trials": trials
        }, f, indent=4)

    print(f"\nPhase 2 complete.")
    print(f"  Best Accuracy : {study.best_value:.2f}%")
    print(f"  Best Params   : {study.best_params}")
    print(f"  Results -> {out_path}")
    return study


# ============================================================
# PHASE 3: Cross-Dataset Generalization
# ============================================================
def phase_cross_dataset(epochs, batch_size):
    """Train on one sensor source, test on the other."""
    repo_root = os.path.abspath(os.path.join(_PROJECT_ROOT, '..'))
    usrp_dir = os.path.join(repo_root, 'unified_dataset')    # Primary unified
    radar_dir = os.path.join(repo_root, '_radar_staging')     # Radar only

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

    out_path = os.path.join(_PROJECT_ROOT, 'results', 'ablation_cross_dataset.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nPhase 3 complete. Results -> {out_path}")
    return results


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IEEE Ablation Study — 3 Phases (Parallel)")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "backbone", "hparam", "cross"],
                        help="Which ablation phase to run")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of Optuna trials for hparam phase")
    parser.add_argument("--max_workers", type=int, default=2,
                        help="Max parallel model-training subprocesses (Phase 1). "
                             "Set to 1 for sequential, 2-4 for parallel GPU.")
    args = parser.parse_args()

    if args.phase in ("all", "backbone"):
        phase_backbone(args.data_dir, args.epochs, args.batch_size, args.max_workers)
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

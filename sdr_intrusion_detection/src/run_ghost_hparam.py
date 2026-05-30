"""
run_ghost_hparam.py — Optuna Hyperparameter Tuning for CGA-Net (Ghost-CAS Full)
===============================================================================
Runs an Optuna hyperparameter sweep specifically on the proposed CGA-Net model
to determine the absolute best training configuration.

Usage:
    python -m src.run_ghost_hparam
    python -m src.run_ghost_hparam --n_trials 20 --epochs 30
"""
import argparse
import json
import os
import optuna

from src.train import train_model
from src.config import (
    DEFAULT_DATA_DIR, RESULTS_DIR, ABLATION_N_TRIALS, DEFAULT_EPOCHS,
    OPTUNA_LR_CHOICES, OPTUNA_BATCH_CHOICES, OPTUNA_OPTIMIZER_CHOICES,
    OPTUNA_GAMMA_CHOICES, OPTUNA_WD_CHOICES, OPTUNA_DROPOUT_CHOICES,
    DEFAULT_PATIENCE
)

# Target model for the hyperparameter search
TARGET_MODEL = 'SDR_GhostCAS_Full'


def _optuna_objective(trial, data_dir, epochs):
    """Single Optuna trial — tune CGA-Net (Ghost-CAS Full)."""
    lr         = trial.suggest_categorical('lr', OPTUNA_LR_CHOICES)
    batch_size = trial.suggest_categorical('batch_size', OPTUNA_BATCH_CHOICES)
    optimizer  = trial.suggest_categorical('optimizer', OPTUNA_OPTIMIZER_CHOICES)
    gamma      = trial.suggest_categorical('focal_gamma', OPTUNA_GAMMA_CHOICES)
    wd         = trial.suggest_categorical('weight_decay', OPTUNA_WD_CHOICES)
    dropout    = trial.suggest_categorical('dropout', OPTUNA_DROPOUT_CHOICES)

    args = argparse.Namespace(
        data_dir=data_dir,
        model=TARGET_MODEL,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        focal_gamma=gamma,
        optimizer=optimizer,
        weight_decay=wd,
        dropout=dropout,
        patience=DEFAULT_PATIENCE
    )

    try:
        log = train_model(args)
        return log["best_val_acc"]
    except Exception as e:
        print(f"  [ERROR] Trial failed: {e}")
        # Return 0 so Optuna avoids this space, but doesn't crash the whole study
        return 0.0


def run_ghost_hparam(args):
    print("=" * 80)
    print(f"  CGA-NET HYPERPARAMETER ABLATION ({args.n_trials} Optuna trials)")
    print("=" * 80)
    print(f"  Target Model : {TARGET_MODEL}")
    print(f"  Epochs / max : {args.epochs}")
    print(f"  Dataset Dir  : {args.data_dir}")
    print("=" * 80)

    study = optuna.create_study(
        direction='maximize',
        study_name='CGA-Net_HPO',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(
        lambda trial: _optuna_objective(trial, args.data_dir, args.epochs),
        n_trials=args.n_trials,
        show_progress_bar=True
    )

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    trials = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            trials.append({
                "number": t.number,
                "value": t.value,
                "params": t.params
            })

    out_path = os.path.join(RESULTS_DIR, 'ghost_ablation_hparam.json')
    with open(out_path, 'w') as f:
        json.dump({
            "best_trial": study.best_trial.number if study.best_trials else None,
            "best_value": study.best_value if study.best_trials else 0.0,
            "best_params": study.best_params if study.best_trials else {},
            "all_trials": trials
        }, f, indent=4)

    print(f"\nCGA-Net Hyperparameter Sweep Complete.")
    if study.best_trials:
        print(f"  Best Accuracy : {study.best_value:.2f}%")
        print(f"  Best Params   : {study.best_params}")
    print(f"  Results saved -> {out_path}")
    
    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Optuna Hyperparameter Sweep specifically on CGA-Net"
    )
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="Max epochs per trial (early stopping handles convergence)")
    parser.add_argument("--n_trials", type=int, default=ABLATION_N_TRIALS,
                        help="Number of Optuna trials to run")
    args = parser.parse_args()
    
    run_ghost_hparam(args)

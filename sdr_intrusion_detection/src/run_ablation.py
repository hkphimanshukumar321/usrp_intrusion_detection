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
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import optuna
from src.train import train_model
from src.evaluate import evaluate_model
from src.config import (
    TIMM_MODEL_MAP, CUSTOM_MODEL_NAME, DEFAULT_DATA_DIR, PROJECT_ROOT,
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LR, DEFAULT_FOCAL_GAMMA,
    DEFAULT_OPTIMIZER, DEFAULT_WEIGHT_DECAY, DEFAULT_DROPOUT,
    ABLATION_MAX_WORKERS, ABLATION_N_TRIALS, ABLATION_SUBPROCESS_TIMEOUT,
    ABLATION_HPARAM_EPOCHS, ABLATION_CROSS_EPOCHS,
    OPTUNA_LR_CHOICES, OPTUNA_BATCH_CHOICES, OPTUNA_OPTIMIZER_CHOICES,
    OPTUNA_GAMMA_CHOICES, OPTUNA_WD_CHOICES, OPTUNA_DROPOUT_CHOICES,
    CROSS_DATASET_UNIFIED, CROSS_DATASET_RADAR,
    RESULTS_DIR, LOGS_DIR,
)

CUSTOM_MODEL = CUSTOM_MODEL_NAME
DEFAULT_DATA = DEFAULT_DATA_DIR

# Root of the cloned repository (sdr_intrusion_detection/)
_PROJECT_ROOT = PROJECT_ROOT


# ============================================================
# PHASE 1: Backbone Comparison  (PARALLEL)
# ============================================================
def _train_single_model_subprocess(model_name, data_dir, epochs, batch_size, lr):
    """
    Train ONE model in a fresh subprocess.
    Streams output line-by-line to both a log file AND stdout (prefixed with model name).
    Returns (model_name, result_dict) or (model_name, error_string).
    """
    cmd = [
        sys.executable, '-u', '-m', 'src.train',   # -u = unbuffered Python output
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
            proc = subprocess.Popen(
                cmd,
                cwd=_PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1  # Line-buffered
            )
            # Stream output line-by-line: write to log + print with prefix
            for line in proc.stdout:
                line = line.rstrip('\n')
                logf.write(line + '\n')
                logf.flush()
                # Print key lines to parent stdout with model tag
                if any(kw in line for kw in ['Epoch', 'Val Acc', 'Best Model', 'PROFILING',
                                              'Train Loss', 'Val Loss', 'Dataset Class',
                                              'Loaded Dataset', 'device:', 'Parameters']):
                    print(f"  [{model_name}] {line}", flush=True)
            proc.wait(timeout=ABLATION_SUBPROCESS_TIMEOUT)

        # After training, read the JSON history produced by train.py
        history_path = os.path.join(LOGS_DIR, f'history_{model_name}.json')
        if os.path.isfile(history_path):
            with open(history_path) as f:
                result = json.load(f)
            return (model_name, result)
        else:
            return (model_name, f"Process exited with code {proc.returncode} but no history JSON found. Check {log_path}")
    except subprocess.TimeoutExpired:
        proc.kill()
        return (model_name, f"TIMEOUT after 2 hours. Check {log_path}")
    except Exception as e:
        return (model_name, str(e))


def phase_backbone(data_dir, epochs, batch_size, max_workers=ABLATION_MAX_WORKERS):
    """Train all 19 architectures using parallel subprocesses."""
    models = [CUSTOM_MODEL] + list(TIMM_MODEL_MAP.keys())
    total = len(models)

    print("=" * 80, flush=True)
    print(f"  PHASE 1: BACKBONE COMPARISON ({total} architectures)", flush=True)
    print(f"  Parallelism: max_workers = {max_workers}", flush=True)
    print("=" * 80, flush=True)

    summary = {}
    failed = []
    completed = 0
    lr = 1e-3
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _train_single_model_subprocess,
                name, data_dir, epochs, batch_size, lr
            ): name
            for name in models
        }

        print(f"\n  ⏳ Submitted {total} models. Waiting for results...\n", flush=True)

        for future in as_completed(futures):
            model_name = futures[future]
            completed += 1
            elapsed = time.time() - t_start
            try:
                name, result = future.result()
                if isinstance(result, dict):
                    summary[name] = {
                        "best_val_acc": result.get("best_val_acc", 0),
                        "profile": result.get("profile", {})
                    }
                    print(f"\n  ✓ [{completed}/{total}] {name} — Val Acc: {result.get('best_val_acc', 0):.2f}%  (elapsed: {elapsed/60:.1f} min)", flush=True)
                else:
                    failed.append(name)
                    print(f"\n  ✗ [{completed}/{total}] {name} FAILED: {result}  (elapsed: {elapsed/60:.1f} min)", flush=True)
            except Exception as exc:
                failed.append(model_name)
                print(f"\n  ✗ [{completed}/{total}] {model_name} Exception: {exc}  (elapsed: {elapsed/60:.1f} min)", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'ablation_backbone.json')
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
    lr         = trial.suggest_categorical('lr', OPTUNA_LR_CHOICES)
    batch_size = trial.suggest_categorical('batch_size', OPTUNA_BATCH_CHOICES)
    optimizer  = trial.suggest_categorical('optimizer', OPTUNA_OPTIMIZER_CHOICES)
    gamma      = trial.suggest_categorical('focal_gamma', OPTUNA_GAMMA_CHOICES)
    wd         = trial.suggest_categorical('weight_decay', OPTUNA_WD_CHOICES)
    dropout    = trial.suggest_categorical('dropout', OPTUNA_DROPOUT_CHOICES)

    args = argparse.Namespace(
        data_dir=data_dir, model=CUSTOM_MODEL,
        epochs=epochs, batch_size=batch_size,
        lr=lr, focal_gamma=gamma, optimizer=optimizer,
        weight_decay=wd, dropout=dropout
    )

    log = train_model(args)
    return log["best_val_acc"]


def phase_hparam(data_dir, epochs, n_trials=ABLATION_N_TRIALS):
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
    os.makedirs(RESULTS_DIR, exist_ok=True)
    trials = []
    for t in study.trials:
        trials.append({
            "number": t.number,
            "value": t.value,
            "params": t.params
        })

    out_path = os.path.join(RESULTS_DIR, 'ablation_hparam.json')
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
def _create_pseudo_split(source_dir, output_dir, seed=42):
    """
    Split the unified dataset into two pseudo-sources (Source_A / Source_B)
    by randomly partitioning each class's files 50/50.

    Creates two ImageFolder-compatible directories:
        output_dir/Source_A/{train,val,test}/{class}/*
        output_dir/Source_B/{train,val,test}/{class}/*

    Uses file copies (not moves) so the original dataset is untouched.
    Returns (path_A, path_B).
    """
    rng = random.Random(seed)
    src_a = os.path.join(output_dir, 'Source_A')
    src_b = os.path.join(output_dir, 'Source_B')

    for split in ('train', 'val', 'test'):
        split_dir = os.path.join(source_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for cls_name in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue

            files = sorted(os.listdir(cls_dir))
            rng.shuffle(files)
            mid = len(files) // 2

            dest_a = os.path.join(src_a, split, cls_name)
            dest_b = os.path.join(src_b, split, cls_name)
            os.makedirs(dest_a, exist_ok=True)
            os.makedirs(dest_b, exist_ok=True)

            for f in files[:mid]:
                shutil.copy2(os.path.join(cls_dir, f), os.path.join(dest_a, f))
            for f in files[mid:]:
                shutil.copy2(os.path.join(cls_dir, f), os.path.join(dest_b, f))

    return src_a, src_b


def phase_cross_dataset(epochs, batch_size):
    """Train on one sensor source, test on the other.

    If two separate dataset directories exist (unified + radar), use them
    directly.  Otherwise, auto-split the unified dataset into two
    pseudo-sources so the experiment can still run.
    """
    repo_root = os.path.abspath(os.path.join(_PROJECT_ROOT, '..'))
    usrp_dir = os.path.join(repo_root, CROSS_DATASET_UNIFIED)
    radar_dir = os.path.join(repo_root, CROSS_DATASET_RADAR)

    pairs = []
    if os.path.isdir(usrp_dir):
        pairs.append(("Unified", usrp_dir))
    if os.path.isdir(radar_dir):
        pairs.append(("Radar", radar_dir))

    cleanup_dir = None  # Will hold temp path if we auto-split

    if len(pairs) < 2:
        # ---------- AUTO-SPLIT FALLBACK ----------
        if len(pairs) == 0:
            print("  [SKIP] No dataset directories found for cross-dataset test.")
            return {}

        base_dir = pairs[0][1]  # The one directory that exists
        print("  [INFO] Only one dataset found. Auto-splitting into two pseudo-sources...")

        split_root = os.path.join(_PROJECT_ROOT, '_cross_dataset_tmp')
        if os.path.isdir(split_root):
            shutil.rmtree(split_root)  # Clean previous run

        src_a, src_b = _create_pseudo_split(base_dir, split_root)

        # Count files for logging
        a_count = sum(len(fs) for _, _, fs in os.walk(src_a))
        b_count = sum(len(fs) for _, _, fs in os.walk(src_b))
        print(f"  [OK] Source_A: {a_count} files  |  Source_B: {b_count} files")

        pairs = [("Source_A", src_a), ("Source_B", src_b)]
        cleanup_dir = split_root

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
                lr=DEFAULT_LR, focal_gamma=DEFAULT_FOCAL_GAMMA,
                optimizer=DEFAULT_OPTIMIZER,
                weight_decay=DEFAULT_WEIGHT_DECAY, dropout=DEFAULT_DROPOUT
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

    # Cleanup temporary pseudo-split
    if cleanup_dir and os.path.isdir(cleanup_dir):
        shutil.rmtree(cleanup_dir)
        print("  [OK] Cleaned up temporary pseudo-split directory.")

    out_path = os.path.join(RESULTS_DIR, 'ablation_cross_dataset.json')
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
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "backbone", "hparam", "cross"],
                        help="Which ablation phase to run")
    parser.add_argument("--n_trials", type=int, default=ABLATION_N_TRIALS,
                        help="Number of Optuna trials for hparam phase")
    parser.add_argument("--max_workers", type=int, default=ABLATION_MAX_WORKERS,
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

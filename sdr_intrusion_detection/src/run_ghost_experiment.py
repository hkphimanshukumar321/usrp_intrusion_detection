"""
run_ghost_experiment.py — Ghost-CAS Architecture Comparison (Full Pipeline)
============================================================================
Trains and compares model variants with COMPLETE evaluation:
  ✓ Model profiling  (params, size, FLOPs, memory, inference latency)
  ✓ Full training    (with early stopping, W&B logging)
  ✓ Confusion matrix (per-class precision/recall/F1)
  ✓ Training curves  (loss + accuracy plots)
  ✓ Final summary    (JSON + printed table)

Usage:
    python -m src.run_ghost_experiment
    python -m src.run_ghost_experiment --epochs 50 --batch_size 32
    python -m src.run_ghost_experiment --models SDR_GhostCAS_Full  # single model
"""
import argparse
import json
import os
import time
import torch
import numpy as np

from src.train import train_model, profile_model
from src.evaluate import evaluate_model
from src.generate_figures import plot_training_curves
from src.model import get_model
from src.config import (
    GHOST_EXPERIMENT_MODELS, DEFAULT_DATA_DIR, DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE, DEFAULT_LR, DEFAULT_FOCAL_GAMMA,
    DEFAULT_OPTIMIZER, DEFAULT_WEIGHT_DECAY, DEFAULT_DROPOUT,
    DEFAULT_PATIENCE, RESULTS_DIR, FIGURES_DIR, IMAGE_SIZE,
    IMAGE_CHANNELS,
)


# ============================================================
# FLOPs + PEAK MEMORY ESTIMATION
# ============================================================
def estimate_flops_and_memory(model, device):
    """
    Estimate FLOPs via torch.profiler and peak GPU/CPU memory usage.
    Falls back to a manual param-based estimate if profiler is unavailable.
    """
    model.to(device).eval()
    dummy = torch.randn(1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)

    # --- Peak Memory ---
    peak_memory_mb = 0.0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            model(dummy)
        peak_memory_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
    else:
        # CPU: estimate from model size + activations (rough)
        model_mb = sum(p.numel() * 4 for p in model.parameters()) / (1024 ** 2)
        peak_memory_mb = round(model_mb * 2.5, 2)  # ~2.5x model size for activations

    # --- FLOPs via torch.profiler ---
    flops = 0
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            with_flops=True,
        ) as prof:
            with torch.no_grad():
                model(dummy)

        # Sum all FLOPs from profiler events
        for event in prof.key_averages():
            if event.flops and event.flops > 0:
                flops += event.flops
    except Exception:
        # Fallback: rough estimate (2 * MACs ≈ 2 * params * spatial_ops)
        total_params = sum(p.numel() for p in model.parameters())
        flops = total_params * 2  # Very rough lower bound

    flops_m = round(flops / 1e6, 2)    # MFLOPs
    flops_g = round(flops / 1e9, 2)    # GFLOPs

    return {
        "FLOPs": flops,
        "FLOPs_M": flops_m,
        "FLOPs_G": flops_g,
        "Peak_Memory_MB": peak_memory_mb,
    }


# ============================================================
# COMPREHENSIVE MODEL PROFILING
# ============================================================
def profile_all_models(model_names):
    """Profile each model: params, size, latency, FLOPs, memory."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    profiles = {}

    print("\n" + "=" * 90)
    print("  MODEL ARCHITECTURE COMPARISON (Comprehensive Profile)")
    print("=" * 90)
    header = (f"  {'Model':<30s}  {'Params':>12s}  {'Size MB':>8s}  "
              f"{'FLOPs G':>8s}  {'Mem MB':>8s}  {'Latency ms':>11s}")
    print(header)
    print("-" * 90)

    for name in model_names:
        model = get_model(name)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        size_mb = round(total * 4 / (1024 ** 2), 2)

        # Inference latency
        prof = profile_model(model, device)
        lat = prof["Inference_Time_ms"]

        # FLOPs + memory
        flops_mem = estimate_flops_and_memory(model, device)

        profiles[name] = {
            "Total_Parameters": total,
            "Trainable_Parameters": trainable,
            "Model_Size_MB": size_mb,
            "Inference_Time_ms": lat,
            **flops_mem,
        }

        print(f"  {name:<30s}  {total:>12,}  {size_mb:>8.2f}  "
              f"{flops_mem['FLOPs_G']:>8.2f}  {flops_mem['Peak_Memory_MB']:>8.2f}  "
              f"{lat:>11.2f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("=" * 90)
    return profiles


# ============================================================
# MAIN EXPERIMENT
# ============================================================
def run_experiment(args):
    """Full experiment: profile → train → evaluate → figures → summary."""
    model_names = args.models if args.models else GHOST_EXPERIMENT_MODELS

    # ---- Phase 1: Comprehensive Profile ----
    profiles = profile_all_models(model_names)

    # ---- Phase 2: Training ----
    print("\n" + "=" * 70)
    print("  GHOST-CAS EXPERIMENT: FULL TRAINING")
    print(f"  Epochs: {args.epochs}  |  Batch Size: {args.batch_size}  |  LR: {args.lr}")
    print("=" * 70)

    all_results = {}
    t_total = time.time()
    trained_models = []   # Track successfully trained models for figure generation

    for i, name in enumerate(model_names, 1):
        print(f"\n{'#' * 60}")
        print(f"  [{i}/{len(model_names)}] Training: {name}")
        print(f"{'#' * 60}\n")

        train_args = argparse.Namespace(
            data_dir=args.data_dir,
            model=name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            focal_gamma=DEFAULT_FOCAL_GAMMA,
            optimizer=DEFAULT_OPTIMIZER,
            weight_decay=DEFAULT_WEIGHT_DECAY,
            dropout=DEFAULT_DROPOUT,
            patience=DEFAULT_PATIENCE,
        )

        t0 = time.time()
        try:
            log = train_model(train_args)
            elapsed = time.time() - t0

            all_results[name] = {
                "best_val_acc": log["best_val_acc"],
                "early_stopped": log.get("early_stopped", False),
                "actual_epochs": log.get("actual_epochs", args.epochs),
                "training_time_min": round(elapsed / 60, 1),
                "profile": profiles.get(name, {}),
            }
            trained_models.append(name)
            print(f"\n  ✓ {name} — Val Acc: {log['best_val_acc']:.2f}%  "
                  f"({elapsed/60:.1f} min)")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n  ✗ {name} FAILED after {elapsed/60:.1f} min: {e}")
            all_results[name] = {"error": str(e)}

    # ---- Phase 3: Evaluation (Confusion Matrix + Classification Report) ----
    print("\n" + "=" * 70)
    print("  PHASE 3: TEST SET EVALUATION (Confusion Matrices)")
    print("=" * 70)

    for name in trained_models:
        print(f"\n  >>> Evaluating {name} on test set...")
        try:
            eval_args = argparse.Namespace(
                data_dir=args.data_dir,
                model=name,
                batch_size=args.batch_size,
            )
            evaluate_model(eval_args)
            print(f"  ✓ Confusion matrix saved for {name}")
        except Exception as e:
            print(f"  ✗ Evaluation FAILED for {name}: {e}")

    # ---- Phase 4: Training Curve Figures ----
    print("\n" + "=" * 70)
    print("  PHASE 4: GENERATING TRAINING CURVES")
    print("=" * 70)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    for name in trained_models:
        try:
            plot_training_curves(name)
            print(f"  ✓ Training curves plotted for {name}")
        except Exception as e:
            print(f"  ✗ Plotting FAILED for {name}: {e}")

    total_time = (time.time() - t_total) / 60

    # ---- Phase 5: Final Summary ----
    print("\n\n" + "=" * 100)
    print("  GHOST-CAS EXPERIMENT — FINAL RESULTS")
    print("=" * 100)
    header = (f"  {'Model':<30s}  {'Val Acc':>8s}  {'Params':>12s}  "
              f"{'Size MB':>8s}  {'FLOPs G':>8s}  {'Mem MB':>8s}  "
              f"{'Lat ms':>7s}  {'Time':>7s}")
    print(header)
    print("-" * 100)

    for name, res in all_results.items():
        if "error" in res:
            print(f"  {name:<30s}  {'FAILED':>8s}")
            continue
        prof = res.get("profile", {})
        print(f"  {name:<30s}  {res['best_val_acc']:>7.2f}%  "
              f"{prof.get('Total_Parameters', 0):>12,}  "
              f"{prof.get('Model_Size_MB', 0):>8.2f}  "
              f"{prof.get('FLOPs_G', 0):>8.2f}  "
              f"{prof.get('Peak_Memory_MB', 0):>8.2f}  "
              f"{prof.get('Inference_Time_ms', 0):>7.2f}  "
              f"{res['training_time_min']:>6.1f}m")

    print("-" * 100)
    print(f"  Total experiment time: {total_time:.1f} minutes")
    print("=" * 100)

    # ---- Save Complete Results ----
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'ghost_experiment_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"\n  Results JSON        → {out_path}")
    print(f"  Confusion matrices  → {RESULTS_DIR}/confusion_matrix_*.png")
    print(f"  Training curves     → {FIGURES_DIR}/training_curves_*.png")
    print(f"  Model checkpoints   → checkpoints/best_*.pth")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ghost-CAS Architecture Experiment — compare model variants "
                    "(profiles, trains, evaluates, generates figures)"
    )
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific model(s) to run. Default: all 3 variants. "
                             "E.g.: --models SDR_GhostCAS_Full")
    args = parser.parse_args()
    run_experiment(args)

"""
run_ghost_eval_only.py — Evaluate Ghost-CAS models from existing checkpoints
==============================================================================
Skips training entirely. Uses checkpoints already saved in checkpoints/
to generate:
  ✓ Model profiling  (params, size, FLOPs, memory, inference latency)
  ✓ Confusion matrix + classification report  (per-class P/R/F1)
  ✓ Training curves  (loss + accuracy from saved history JSONs)
  ✓ Final summary table

Usage:
    python -m src.run_ghost_eval_only
    python -m src.run_ghost_eval_only --models SDR_GhostCAS_Full SDR_GhostCAS_ResNet
    python -m src.run_ghost_eval_only --models SDR_Custom_CoordASPP_Focal
"""
import argparse
import json
import os
import glob
import torch

from src.train import profile_model
from src.evaluate import evaluate_model
from src.generate_figures import plot_training_curves
from src.model import get_model
from src.config import (
    GHOST_EXPERIMENT_MODELS, DEFAULT_DATA_DIR, DEFAULT_BATCH_SIZE,
    RESULTS_DIR, FIGURES_DIR, LOGS_DIR, CHECKPOINT_DIR,
    IMAGE_SIZE, IMAGE_CHANNELS,
)


# ============================================================
# FLOPs + PEAK MEMORY ESTIMATION
# ============================================================
def estimate_flops_and_memory(model, device):
    """Estimate FLOPs via torch.profiler and peak GPU/CPU memory."""
    model.to(device).eval()
    dummy = torch.randn(1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)

    peak_memory_mb = 0.0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            model(dummy)
        peak_memory_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
    else:
        model_mb = sum(p.numel() * 4 for p in model.parameters()) / (1024 ** 2)
        peak_memory_mb = round(model_mb * 2.5, 2)

    flops = 0
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            with_flops=True,
        ) as prof:
            with torch.no_grad():
                model(dummy)
        for event in prof.key_averages():
            if event.flops and event.flops > 0:
                flops += event.flops
    except Exception:
        total_params = sum(p.numel() for p in model.parameters())
        flops = total_params * 2

    return {
        "FLOPs": flops,
        "FLOPs_M": round(flops / 1e6, 2),
        "FLOPs_G": round(flops / 1e9, 2),
        "Peak_Memory_MB": peak_memory_mb,
    }


# ============================================================
# AUTO-DETECT AVAILABLE CHECKPOINTS
# ============================================================
def discover_checkpoints(requested_models=None):
    """Find all best_*.pth checkpoints and return matching model names."""
    available = []
    if not os.path.isdir(CHECKPOINT_DIR):
        print(f"  [ERROR] Checkpoint directory not found: {CHECKPOINT_DIR}")
        return []

    ckpt_files = glob.glob(os.path.join(CHECKPOINT_DIR, 'best_*.pth'))
    for f in ckpt_files:
        basename = os.path.basename(f)
        # Extract model name: best_<MODEL_NAME>.pth
        model_name = basename.replace('best_', '').replace('.pth', '')
        available.append(model_name)

    if requested_models:
        # Filter to only requested models that have checkpoints
        found = []
        for m in requested_models:
            if m in available:
                found.append(m)
            else:
                print(f"  [WARN] Checkpoint not found for '{m}' — skipping")
        return found

    return available


# ============================================================
# MAIN EVALUATION PIPELINE
# ============================================================
def run_eval_only(args):
    """Run evaluation pipeline using existing checkpoints (no training)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Discover which models have checkpoints
    if args.models:
        model_names = args.models
    else:
        model_names = discover_checkpoints()

    if not model_names:
        print("  [ERROR] No checkpoints found! Train models first.")
        return

    print("\n" + "=" * 90)
    print("  GHOST-CAS EVALUATION ONLY (From Existing Checkpoints)")
    print("=" * 90)
    print(f"  Device     : {device}")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    print(f"  Models     : {', '.join(model_names)}")
    print("=" * 90)

    # ---- Phase 1: Comprehensive Profile ----
    print("\n" + "=" * 90)
    print("  PHASE 1: MODEL PROFILING")
    print("=" * 90)
    header = (f"  {'Model':<30s}  {'Params':>12s}  {'Size MB':>8s}  "
              f"{'FLOPs G':>8s}  {'Mem MB':>8s}  {'Latency ms':>11s}")
    print(header)
    print("-" * 90)

    profiles = {}
    for name in model_names:
        try:
            model = get_model(name)
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            size_mb = round(total * 4 / (1024 ** 2), 2)

            prof = profile_model(model, device)
            lat = prof["Inference_Time_ms"]
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
        except Exception as e:
            print(f"  {name:<30s}  PROFILE FAILED: {e}")

    print("=" * 90)

    # ---- Phase 2: Confusion Matrix + Classification Report ----
    print("\n" + "=" * 70)
    print("  PHASE 2: TEST SET EVALUATION (Confusion Matrices)")
    print("=" * 70)

    evaluated = []
    for name in model_names:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'best_{name}.pth')
        if not os.path.exists(ckpt_path):
            print(f"\n  [SKIP] No checkpoint for {name} at {ckpt_path}")
            continue

        print(f"\n  >>> Evaluating {name} on test set...")
        try:
            eval_args = argparse.Namespace(
                data_dir=args.data_dir,
                model=name,
                batch_size=args.batch_size,
            )
            evaluate_model(eval_args)
            evaluated.append(name)
            print(f"  ✓ Confusion matrix + classification report saved for {name}")
        except Exception as e:
            print(f"  ✗ Evaluation FAILED for {name}: {e}")

    # ---- Phase 3: Training Curve Figures ----
    print("\n" + "=" * 70)
    print("  PHASE 3: GENERATING TRAINING CURVES")
    print("=" * 70)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    for name in model_names:
        history_path = os.path.join(LOGS_DIR, f'history_{name}.json')
        if not os.path.exists(history_path):
            print(f"  [SKIP] No training history for {name}")
            continue
        try:
            plot_training_curves(name)
            print(f"  ✓ Training curves plotted for {name}")
        except Exception as e:
            print(f"  ✗ Plotting FAILED for {name}: {e}")

    # ---- Phase 4: Summary ----
    print("\n\n" + "=" * 100)
    print("  EVALUATION SUMMARY")
    print("=" * 100)
    header = (f"  {'Model':<30s}  {'Params':>12s}  {'Size MB':>8s}  "
              f"{'FLOPs G':>8s}  {'Mem MB':>8s}  {'Lat ms':>7s}  "
              f"{'Checkpoint':>12s}")
    print(header)
    print("-" * 100)

    for name in model_names:
        prof = profiles.get(name, {})
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'best_{name}.pth')
        ckpt_size = ""
        if os.path.exists(ckpt_path):
            ckpt_size = f"{os.path.getsize(ckpt_path) / (1024**2):.1f} MB"
        else:
            ckpt_size = "MISSING"

        print(f"  {name:<30s}  "
              f"{prof.get('Total_Parameters', 0):>12,}  "
              f"{prof.get('Model_Size_MB', 0):>8.2f}  "
              f"{prof.get('FLOPs_G', 0):>8.2f}  "
              f"{prof.get('Peak_Memory_MB', 0):>8.2f}  "
              f"{prof.get('Inference_Time_ms', 0):>7.2f}  "
              f"{ckpt_size:>12s}")

    print("=" * 100)

    # Save summary
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = {"profiles": profiles, "evaluated_models": evaluated}

    # Merge training history best_val_acc if available
    for name in model_names:
        history_path = os.path.join(LOGS_DIR, f'history_{name}.json')
        if os.path.exists(history_path):
            with open(history_path) as f:
                hist = json.load(f)
            summary.setdefault("training_results", {})[name] = {
                "best_val_acc": hist.get("best_val_acc", "N/A"),
                "actual_epochs": hist.get("actual_epochs", "N/A"),
            }

    out_path = os.path.join(RESULTS_DIR, 'ghost_eval_results.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\n  Results JSON        → {out_path}")
    print(f"  Confusion matrices  → {RESULTS_DIR}/confusion_matrix_*.png")
    print(f"  Training curves     → {FIGURES_DIR}/training_curves_*.png")
    print(f"  Checkpoints used    → {CHECKPOINT_DIR}/best_*.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Ghost-CAS models from existing checkpoints "
                    "(no training — just profile, confusion matrix, curves)"
    )
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model(s) to evaluate. Default: auto-detect from checkpoints/. "
                             "E.g.: --models SDR_GhostCAS_Full SDR_Custom_CoordASPP_Focal")
    args = parser.parse_args()
    run_eval_only(args)

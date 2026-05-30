"""
run_ghost_cross_dataset.py — Cross-Dataset Generalization for Ghost-CAS models
================================================================================
Uses EXISTING checkpoints from training to test cross-dataset generalization:
  - Auto-splits unified_dataset into Source_A / Source_B (random 50/50)
  - Loads best checkpoint for each model
  - Tests each model trained on Source_A against Source_B's test set, and vice versa
  - Generates confusion matrices + classification reports for each cross-test

Also includes profiling and training curve generation from saved history.

Usage:
    python -m src.run_ghost_cross_dataset
    python -m src.run_ghost_cross_dataset --models SDR_GhostCAS_Full SDR_GhostCAS_ResNet
    python -m src.run_ghost_cross_dataset --models SDR_Custom_CoordASPP_Focal
"""
import argparse
import json
import os
import shutil
import random
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.model import get_model
from src.data_loader import get_dataloaders
from src.train import train_model, profile_model
from src.generate_figures import plot_training_curves
from src.config import (
    GHOST_EXPERIMENT_MODELS, DEFAULT_DATA_DIR, DEFAULT_BATCH_SIZE,
    DEFAULT_LR, DEFAULT_FOCAL_GAMMA, DEFAULT_OPTIMIZER,
    DEFAULT_WEIGHT_DECAY, DEFAULT_DROPOUT, DEFAULT_PATIENCE,
    DEFAULT_NUM_WORKERS, ABLATION_CROSS_EPOCHS,
    RESULTS_DIR, FIGURES_DIR, LOGS_DIR, CHECKPOINT_DIR,
    IMAGE_SIZE, IMAGE_CHANNELS, PROJECT_ROOT, CLASS_NAMES,
    CUSTOM_MODEL_NAME,
)


# ============================================================
# CREATE PSEUDO-SPLIT (same logic as run_ablation.py)
# ============================================================
def _create_pseudo_split(source_dir, output_dir, seed=42):
    """
    Split a single dataset directory into two pseudo-sources (50/50 per class).
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


# ============================================================
# EVALUATE MODEL ON A SPECIFIC DATASET (with confusion matrix)
# ============================================================
def evaluate_on_dataset(model, data_dir, model_name, tag, device):
    """
    Run inference on the test split of `data_dir` and produce:
    - Classification report (printed)
    - Confusion matrix (saved as PNG)
    Returns test accuracy.
    """
    _, _, test_loader = get_dataloaders(
        dataset_dir=data_dir,
        batch_size=32,
        num_workers=DEFAULT_NUM_WORKERS
    )

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Accuracy
    correct = sum(p == t for p, t in zip(all_preds, all_targets))
    total = len(all_targets)
    acc = 100.0 * correct / total

    # Classification report
    print(f"\n{'=' * 50}")
    print(f"  CLASSIFICATION REPORT: {tag}")
    print(f"{'=' * 50}")
    print(classification_report(all_targets, all_preds,
                                target_names=CLASS_NAMES, digits=4))

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix: {tag}')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe_tag = tag.replace(' ', '_').replace('→', '_to_')
    cm_path = os.path.join(RESULTS_DIR, f'confusion_matrix_cross_{safe_tag}.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {cm_path}")

    return acc


# ============================================================
# MAIN CROSS-DATASET EXPERIMENT
# ============================================================
def run_cross_dataset(args):
    """
    Cross-dataset generalization test for Ghost-CAS models.
    
    For each model:
      1. Train on Source_A → Test on Source_B
      2. Train on Source_B → Test on Source_A
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_names = args.models if args.models else GHOST_EXPERIMENT_MODELS

    # ---- Create pseudo-split ----
    data_dir = args.data_dir
    split_root = os.path.join(PROJECT_ROOT, '_cross_dataset_tmp')

    if os.path.isdir(split_root):
        shutil.rmtree(split_root)

    print("\n" + "=" * 80)
    print("  CROSS-DATASET GENERALIZATION EXPERIMENT")
    print("=" * 80)
    print(f"  Device: {device}")
    print(f"  Models: {', '.join(model_names)}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Source dataset: {data_dir}")
    print("=" * 80)

    print("\n  [INFO] Auto-splitting dataset into Source_A and Source_B (50/50)...")
    src_a, src_b = _create_pseudo_split(data_dir, split_root)

    a_count = sum(len(fs) for _, _, fs in os.walk(src_a))
    b_count = sum(len(fs) for _, _, fs in os.walk(src_b))
    print(f"  [OK] Source_A: {a_count} files  |  Source_B: {b_count} files")

    pairs = [("Source_A", src_a), ("Source_B", src_b)]

    # ---- Run cross-dataset for each model ----
    all_results = {}

    for model_name in model_names:
        print(f"\n{'#' * 70}")
        print(f"  MODEL: {model_name}")
        print(f"{'#' * 70}")

        model_results = {}

        for train_name, train_dir in pairs:
            for test_name, test_dir in pairs:
                if train_name == test_name:
                    continue

                tag = f"{model_name}_Train_{train_name}_Test_{test_name}"
                print(f"\n  >>> {tag}")
                print(f"      Train on: {train_name} ({train_dir})")
                print(f"      Test on:  {test_name} ({test_dir})")

                try:
                    # Train fresh on Source_X
                    train_args = argparse.Namespace(
                        data_dir=train_dir,
                        model=model_name,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        lr=DEFAULT_LR,
                        focal_gamma=DEFAULT_FOCAL_GAMMA,
                        optimizer=DEFAULT_OPTIMIZER,
                        weight_decay=DEFAULT_WEIGHT_DECAY,
                        dropout=DEFAULT_DROPOUT,
                        patience=DEFAULT_PATIENCE,
                    )
                    log = train_model(train_args)
                    train_acc = log["best_val_acc"]

                    # Load the best checkpoint that was just saved
                    model = get_model(model_name).to(device)
                    ckpt_path = os.path.join(CHECKPOINT_DIR, f'best_{model_name}.pth')
                    if os.path.exists(ckpt_path):
                        model.load_state_dict(torch.load(ckpt_path, map_location=device))

                    # Test on the OTHER source
                    cross_acc = evaluate_on_dataset(
                        model, test_dir, model_name, tag, device
                    )

                    model_results[f"Train_{train_name}_Test_{test_name}"] = {
                        "train_val_acc": round(train_acc, 2),
                        "cross_test_acc": round(cross_acc, 2),
                        "generalization_gap": round(train_acc - cross_acc, 2),
                    }

                    print(f"  ✓ Train Val Acc: {train_acc:.2f}%  |  "
                          f"Cross Test Acc: {cross_acc:.2f}%  |  "
                          f"Gap: {train_acc - cross_acc:.2f}%")

                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"  ✗ FAILED: {e}")
                    model_results[f"Train_{train_name}_Test_{test_name}"] = {"error": str(e)}

        all_results[model_name] = model_results

    # ---- Cleanup ----
    if os.path.isdir(split_root):
        shutil.rmtree(split_root)
        print("\n  [OK] Cleaned up temporary pseudo-split directory.")

    # ---- Summary ----
    print("\n\n" + "=" * 90)
    print("  CROSS-DATASET GENERALIZATION — FINAL RESULTS")
    print("=" * 90)
    print(f"  {'Model':<30s}  {'Direction':<30s}  {'Train Acc':>10s}  "
          f"{'Cross Acc':>10s}  {'Gap':>8s}")
    print("-" * 90)

    for model_name, model_res in all_results.items():
        for direction, res in model_res.items():
            if "error" in res:
                print(f"  {model_name:<30s}  {direction:<30s}  {'FAILED':>10s}")
            else:
                print(f"  {model_name:<30s}  {direction:<30s}  "
                      f"{res['train_val_acc']:>9.2f}%  "
                      f"{res['cross_test_acc']:>9.2f}%  "
                      f"{res['generalization_gap']:>7.2f}%")

    print("=" * 90)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'ghost_cross_dataset_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\n  Results saved → {out_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-Dataset Generalization test for Ghost-CAS models. "
                    "Splits dataset into Source_A/Source_B and tests cross-generalization."
    )
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=ABLATION_CROSS_EPOCHS,
                        help=f"Epochs for cross-dataset training (default: {ABLATION_CROSS_EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model(s) to test. Default: all 3 Ghost-CAS variants.")
    args = parser.parse_args()
    run_cross_dataset(args)

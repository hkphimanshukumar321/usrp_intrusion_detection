"""
run_ghost_cross_test.py — Cross-Dataset TEST ONLY (no training)
================================================================
Loads EXISTING checkpoints and tests them on random splits of the dataset.

What it does:
  1. Auto-splits unified_dataset into Source_A / Source_B (50/50 random)
  2. Loads each model's best checkpoint from checkpoints/
  3. Tests each model on BOTH Source_A and Source_B test sets
  4. Generates confusion matrices + classification reports + summary

Usage:
    python -m src.run_ghost_cross_test
    python -m src.run_ghost_cross_test --models SDR_GhostCAS_Full SDR_GhostCAS_ResNet
"""
import argparse
import glob
import json
import os
import random
import shutil
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.model import get_model
from src.data_loader import get_dataloaders
from src.config import (
    GHOST_EXPERIMENT_MODELS, DEFAULT_DATA_DIR, DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS, RESULTS_DIR, CHECKPOINT_DIR,
    PROJECT_ROOT, CLASS_NAMES,
)


# ============================================================
# PSEUDO-SPLIT
# ============================================================
def _create_pseudo_split(source_dir, output_dir, seed=42):
    """Split dataset 50/50 into Source_A and Source_B per class."""
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
# TEST A MODEL ON A DATASET SPLIT
# ============================================================
def test_on_split(model, data_dir, model_name, split_name, device):
    """
    Run inference only (no training). Returns accuracy and saves
    confusion matrix + classification report.
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

    correct = sum(p == t for p, t in zip(all_preds, all_targets))
    total = len(all_targets)
    acc = 100.0 * correct / total

    # Classification report
    tag = f"{model_name}_on_{split_name}"
    print(f"\n{'=' * 55}")
    print(f"  {tag}")
    print(f"{'=' * 55}")
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
    cm_path = os.path.join(RESULTS_DIR, f'confusion_matrix_cross_{tag}.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix → {cm_path}")

    return acc


# ============================================================
# MAIN
# ============================================================
def run_cross_test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_names = args.models if args.models else GHOST_EXPERIMENT_MODELS

    # ---- Verify checkpoints exist ----
    valid_models = []
    for name in model_names:
        ckpt = os.path.join(CHECKPOINT_DIR, f'best_{name}.pth')
        if os.path.exists(ckpt):
            valid_models.append(name)
        else:
            print(f"  [SKIP] No checkpoint for {name} at {ckpt}")

    if not valid_models:
        print("  [ERROR] No checkpoints found. Train models first!")
        return

    # ---- Create pseudo-split ----
    split_root = os.path.join(PROJECT_ROOT, '_cross_test_tmp')
    if os.path.isdir(split_root):
        shutil.rmtree(split_root)

    print("\n" + "=" * 80)
    print("  CROSS-DATASET TEST (Inference Only — No Training)")
    print("=" * 80)
    print(f"  Device     : {device}")
    print(f"  Models     : {', '.join(valid_models)}")
    print(f"  Dataset    : {args.data_dir}")

    print("\n  Splitting dataset into Source_A / Source_B (50/50 random)...")
    src_a, src_b = _create_pseudo_split(args.data_dir, split_root)

    a_count = sum(len(fs) for _, _, fs in os.walk(src_a))
    b_count = sum(len(fs) for _, _, fs in os.walk(src_b))
    print(f"  Source_A: {a_count} files  |  Source_B: {b_count} files")
    print("=" * 80)

    splits = [("Source_A", src_a), ("Source_B", src_b)]

    # ---- Test each model on each split ----
    all_results = {}

    for name in valid_models:
        print(f"\n{'#' * 60}")
        print(f"  Loading checkpoint: best_{name}.pth")
        print(f"{'#' * 60}")

        model = get_model(name).to(device)
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'best_{name}.pth')
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"  ✓ Loaded: {ckpt_path}")

        model_results = {}
        for split_name, split_dir in splits:
            print(f"\n  >>> Testing {name} on {split_name}...")
            acc = test_on_split(model, split_dir, name, split_name, device)
            model_results[split_name] = round(acc, 2)
            print(f"  ✓ {name} on {split_name}: {acc:.2f}%")

        # Generalization gap = difference between the two splits
        accs = list(model_results.values())
        model_results["gap"] = round(abs(accs[0] - accs[1]), 2)
        all_results[name] = model_results

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Cleanup ----
    if os.path.isdir(split_root):
        shutil.rmtree(split_root)
        print("\n  [OK] Cleaned up temporary split directory.")

    # ---- Final Summary ----
    print("\n\n" + "=" * 80)
    print("  CROSS-DATASET TEST — FINAL RESULTS")
    print("=" * 80)
    print(f"  {'Model':<30s}  {'Source_A':>10s}  {'Source_B':>10s}  {'Gap':>8s}")
    print("-" * 80)

    for name, res in all_results.items():
        print(f"  {name:<30s}  {res['Source_A']:>9.2f}%  "
              f"{res['Source_B']:>9.2f}%  {res['gap']:>7.2f}%")

    print("=" * 80)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'ghost_cross_test_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\n  Results → {out_path}")
    print(f"  Confusion matrices → {RESULTS_DIR}/confusion_matrix_cross_*.png")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-dataset TEST ONLY — loads existing checkpoints, "
                    "splits dataset randomly, tests on both splits."
    )
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model(s) to test. Default: all Ghost-CAS variants.")
    args = parser.parse_args()
    run_cross_test(args)

#!/usr/bin/env python3
"""
train.py — Training pipeline for all intrusion detection models.

Supports:
  - Single-input models (MLP, CNN1D, CNN2D) and dual-input models (Fusion)
  - 5-fold stratified cross-validation
  - SNR-sweep evaluation
  - Early stopping
  - Model checkpointing
  - SVM and Spike-based baseline training

Usage:
    python -m src.train --model dual_branch_fusion --data_dir dataset/simulated
    python -m src.train --model cnn1d_iq --epochs 50 --batch_size 256
    python -m src.train --train_all   # train all models for ablation comparison
"""

import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from src.data_loader import IQDataset, IQAugmentation, CLASS_NAMES, WINDOW_SIZE
from src.feature_extraction import (
    compute_spectrogram_batch,
    compute_statistical_features_batch,
    MultiRepresentationDataset,
)
from src.spike_detector import SpikeBasedClassifier
from src.model import (
    build_model, is_dual_input, MODEL_REGISTRY,
    DUAL_INPUT_MODELS, print_model_summary,
)


# ============================================================
# Training Configuration
# ============================================================
DEFAULT_CONFIG = {
    'epochs': 50,
    'batch_size': 128,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'label_smoothing': 0.1,
    'patience': 10,
    'n_folds': 5,
    'num_workers': min(4, os.cpu_count() or 1),
    'device': 'auto',
    'seed': 42,
}


def get_device(preference: str = 'auto') -> torch.device:
    if preference == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(preference)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Single Epoch Training
# ============================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    dual_input: bool = False,
) -> Tuple[float, float]:
    """Train for one epoch, return (loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        if dual_input:
            iq = batch['iq'].to(device)
            spec = batch['spectrogram'].to(device)
            labels = batch['label'].to(device)
            logits = model(iq, spec)
        else:
            if isinstance(batch, dict):
                inputs = batch['iq'].to(device)
                labels = batch['label'].to(device)
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

            # Route to correct model input
            if hasattr(model, 'encoder') and isinstance(
                list(model.children())[0], nn.Sequential
            ):
                # Check if model expects spectrogram
                first_conv = None
                for m in model.modules():
                    if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                        first_conv = m
                        break
                if first_conv and isinstance(first_conv, nn.Conv2d):
                    # 2D model expects spectrogram — skip for now
                    continue

            logits = model(inputs)

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


# ============================================================
# Validation
# ============================================================
@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    dual_input: bool = False,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Validate model, return (loss, accuracy, all_preds, all_labels)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        if dual_input:
            iq = batch['iq'].to(device)
            spec = batch['spectrogram'].to(device)
            labels = batch['label'].to(device)
            logits = model(iq, spec)
        else:
            if isinstance(batch, dict):
                inputs = batch['iq'].to(device)
                labels = batch['label'].to(device)
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
            logits = model(inputs)

        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        all_preds.append(predicted.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return avg_loss, accuracy, all_preds, all_labels


# ============================================================
# Full Training Loop with K-Fold CV
# ============================================================
def train_model(
    model_name: str,
    data_dir: str,
    output_dir: str = "results/models",
    config: dict = None,
    dataset: IQDataset = None,
) -> Dict:
    """
    Train a model with k-fold cross-validation.

    Args:
        model_name: Name from MODEL_REGISTRY
        data_dir: Path to dataset directory
        output_dir: Where to save model checkpoints
        config: Training configuration dict

    Returns:
        Dictionary of results (per-fold accuracies, best metrics, etc.)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    set_seed(cfg['seed'])
    device = get_device(cfg['device'])
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU Specs: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB VRAM)")
    print(f"  Config: {json.dumps(cfg, indent=2)}")
    print(f"{'='*60}\n")

    # --- Load dataset ---
    dual_input = is_dual_input(model_name)

    if dataset is not None:
        iq_dataset = dataset
    else:
        # Load fresh if not provided
        iq_dataset = IQDataset(data_dir, window_size=WINDOW_SIZE)

    # IQDataset now returns dicts {'iq', 'spectrogram', 'label'}
    # So we don't need MultiRepresentationDataset anymore.
    dataset = iq_dataset

    # --- K-Fold CV ---
    skf = StratifiedKFold(
        n_splits=cfg['n_folds'], shuffle=True, random_state=cfg['seed']
    )
    labels = iq_dataset.labels

    fold_results = []
    best_overall_acc = 0.0

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels)
    ):
        print(f"\n--- Fold {fold_idx+1}/{cfg['n_folds']} ---")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_subset, batch_size=cfg['batch_size'], shuffle=True,
            num_workers=cfg['num_workers'], pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_subset, batch_size=cfg['batch_size'], shuffle=False,
            num_workers=cfg['num_workers'], pin_memory=True,
        )

        # Build fresh model for each fold
        model = build_model(model_name).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
        optimizer = optim.AdamW(
            model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

        # Training loop with early stopping
        best_val_acc = 0.0
        patience_counter = 0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        for epoch in range(cfg['epochs']):
            t0 = time.time()

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, dual_input
            )
            val_loss, val_acc, val_preds, val_labels = validate(
                model, val_loader, criterion, device, dual_input
            )
            scheduler.step(val_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]['lr']

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{cfg['epochs']} | "
                      f"Train: {train_loss:.4f}/{train_acc:.4f} | "
                      f"Val: {val_loss:.4f}/{val_acc:.4f} | "
                      f"LR: {lr:.2e} | {elapsed:.1f}s")

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model for this fold
                ckpt_path = os.path.join(
                    output_dir, f"{model_name}_fold{fold_idx}.pt"
                )
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'fold': fold_idx,
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'config': cfg,
                }, ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= cfg['patience']:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        fold_results.append({
            'fold': fold_idx,
            'best_val_acc': best_val_acc,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'final_preds': val_preds.tolist(),
            'final_labels': val_labels.tolist(),
        })

        if best_val_acc > best_overall_acc:
            best_overall_acc = best_val_acc

        print(f"  Fold {fold_idx+1} best val accuracy: {best_val_acc:.4f}")

    # --- Aggregate results ---
    fold_accs = [r['best_val_acc'] for r in fold_results]
    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)

    results = {
        'model_name': model_name,
        'num_params': build_model(model_name).count_params(),
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'fold_accuracies': fold_accs,
        'best_accuracy': best_overall_acc,
        'config': cfg,
        'fold_results': fold_results,
    }

    # Save results
    results_path = os.path.join(output_dir, f"{model_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  {model_name} Results:")
    print(f"  Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Best Accuracy: {best_overall_acc:.4f}")
    print(f"  Params: {results['num_params']:,}")
    print(f"  Saved to: {results_path}")
    print(f"{'='*60}\n")

    return results


# ============================================================
# SVM Baseline Training
# ============================================================
def train_svm_baseline(
    data_dir: str,
    output_dir: str = "results/models",
    n_folds: int = 5,
    seed: int = 42,
    dataset: IQDataset = None,
) -> Dict:
    """Train SVM on handcrafted statistical features."""
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Training: SVM Baseline (handcrafted features)")
    print(f"{'='*60}\n")

    # Load data
    if dataset is not None:
        ds = dataset
    else:
        ds = IQDataset(data_dir, window_size=WINDOW_SIZE, max_windows_per_class=2000)
        
    windows = ds.windows     # [N, W, 2]
    labels = ds.labels       # [N]

    # Extract features
    print("  Computing statistical features...")
    features = compute_statistical_features_batch(windows)
    print(f"  Features shape: {features.shape}")

    # K-Fold CV
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accs = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(features, labels)
    ):
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Train SVM
        svm = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=seed)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        fold_accs.append(acc)
        print(f"  Fold {fold_idx+1}: accuracy = {acc:.4f}")

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)

    results = {
        'model_name': 'svm_baseline',
        'num_params': 'N/A (non-parametric)',
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'fold_accuracies': fold_accs,
        'feature_dim': features.shape[1],
    }

    results_path = os.path.join(output_dir, "svm_baseline_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  SVM Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    return results


# ============================================================
# Spike Baseline Training
# ============================================================
def train_spike_baseline(
    data_dir: str,
    output_dir: str = "results/models",
    n_folds: int = 5,
    seed: int = 42,
    dataset: IQDataset = None,
) -> Dict:
    """Train spike-based classifier (non-ML baseline)."""
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Training: Spike Detector Baseline (non-ML)")
    print(f"{'='*60}\n")

    if dataset is not None:
        ds = dataset
    else:
        ds = IQDataset(data_dir, window_size=WINDOW_SIZE,
                       max_windows_per_class=500)
    
    windows = ds.windows
    labels = ds.labels

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accs = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(windows, labels)
    ):
        clf = SpikeBasedClassifier()
        clf.fit(windows[train_idx], labels[train_idx])
        preds = clf.predict(windows[val_idx])
        acc = accuracy_score(labels[val_idx], preds)
        fold_accs.append(acc)
        print(f"  Fold {fold_idx+1}: accuracy = {acc:.4f}")

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)

    results = {
        'model_name': 'spike_baseline',
        'num_params': 0,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'fold_accuracies': fold_accs,
    }

    results_path = os.path.join(output_dir, "spike_baseline_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Spike Baseline Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    return results


# ============================================================
# Train All Models (for ablation study)
def train_all_models(data_dir: str, output_dir: str = "results/models",
                     config: dict = None) -> Dict:
    """Train all models and compile comparison table."""
    all_results = {}
    # 0. Load shared dataset once for all models
    print(f"\n  [SHARED LOAD] Preparing dataset once for all models...")
    # NOTE: We use full dataset for neural nets. Baselines will subsample IQ windows internally.
    shared_dataset = IQDataset(data_dir, window_size=WINDOW_SIZE)

    # 1. Non-ML baselines
    print("\n" + "=" * 60)
    print("  PHASE 1: Non-ML Baselines")
    print("=" * 60)

    all_results['spike_baseline'] = train_spike_baseline(data_dir, output_dir, dataset=shared_dataset)
    all_results['svm_baseline'] = train_svm_baseline(data_dir, output_dir, dataset=shared_dataset)

    # 2. Neural network models
    print("\n" + "=" * 60)
    print("  PHASE 2: Neural Network Models")
    print("=" * 60)

    nn_models = ['mlp_baseline', 'cnn1d_iq', 'cnn2d_spec',
                 'dual_branch_fusion', 'dual_branch_lite']

    for model_name in nn_models:
        try:
            all_results[model_name] = train_model(
                model_name, data_dir, output_dir, config, dataset=shared_dataset
            )
        except Exception as e:
            print(f"  ❌ {model_name} failed: {e}")
            all_results[model_name] = {
                'model_name': model_name,
                'error': str(e),
            }

    # 3. Print comparison table
    print(f"\n{'='*80}")
    print(f"  ABLATION STUDY — Model Comparison")
    print(f"{'='*80}")
    print(f"  {'Model':<25} {'Params':>12} {'Mean Acc':>10} {'± Std':>8} {'Type':>15}")
    print(f"  {'-'*25} {'-'*12} {'-'*10} {'-'*8} {'-'*15}")

    for name, res in all_results.items():
        if 'error' in res:
            print(f"  {name:<25} {'FAILED':>12}")
            continue
        params = res.get('num_params', 'N/A')
        mean = res.get('mean_accuracy', 0)
        std = res.get('std_accuracy', 0)
        model_type = 'Non-ML' if name in ('spike_baseline', 'svm_baseline') else 'Neural'
        params_str = f"{params:,}" if isinstance(params, int) else str(params)
        print(f"  {name:<25} {params_str:>12} {mean:>10.4f} {std:>8.4f} {model_type:>15}")

    print(f"{'='*80}\n")

    # Save combined results
    combined_path = os.path.join(output_dir, "ablation_results.json")
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Combined results saved to: {combined_path}")

    return all_results


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train intrusion detection models"
    )
    parser.add_argument("--model", type=str, default="dual_branch_fusion",
                        choices=list(MODEL_REGISTRY.keys()) + ['svm', 'spike'],
                        help="Model to train")
    parser.add_argument("--data_dir", type=str, default="dataset/simulated",
                        help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="results/models",
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_all", action="store_true",
                        help="Train all models for ablation study")
    parser.add_argument("--show_models", action="store_true",
                        help="Print model architecture summary")

    args = parser.parse_args()

    if args.show_models:
        print_model_summary()
        return

    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'n_folds': args.n_folds,
        'patience': args.patience,
        'seed': args.seed,
    }

    if args.train_all:
        train_all_models(args.data_dir, args.output_dir, config)
    elif args.model == 'svm':
        train_svm_baseline(args.data_dir, args.output_dir)
    elif args.model == 'spike':
        train_spike_baseline(args.data_dir, args.output_dir)
    else:
        train_model(args.model, args.data_dir, args.output_dir, config)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
evaluate.py — Generate all paper-ready figures and tables.

Produces:
  1. Confusion Matrix (4x4 heatmap)
  2. ROC Curves (one-vs-rest with AUC)
  3. Accuracy vs SNR plot
  4. t-SNE feature visualization
  5. Training loss/accuracy curves
  6. Spectrogram gallery (examples per class)
  7. Spike visualization overlay
  8. Ablation comparison bar chart
  9. Edge inference benchmark table

Usage:
    python -m src.evaluate --results_dir results/models --output_dir results/figures
"""

import argparse
import json
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_fscore_support,
)
from sklearn.manifold import TSNE
from typing import Dict, List, Optional

from src.data_loader import CLASS_NAMES, WINDOW_SIZE, IQDataset
from src.feature_extraction import compute_spectrogram
from src.spike_detector import detect_spikes

# ============================================================
# Plot Style Configuration
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

COLORS = ['#2196F3', '#F44336', '#FF9800', '#4CAF50']
CLASS_LABELS = list(CLASS_NAMES.values())


# ============================================================
# 1. Confusion Matrix
# ============================================================
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: str,
    normalize: bool = True,
):
    """Plot and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='.2%' if normalize else 'd',
        cmap='Blues', xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
        ax=ax, cbar_kws={'label': 'Proportion'},
        linewidths=0.5, linecolor='white',
    )
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    ax.set_title(f'Confusion Matrix — {model_name}')
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  ✓ Confusion matrix → {output_path}")


# ============================================================
# 2. ROC Curves
# ============================================================
def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    output_path: str,
):
    """Plot one-vs-rest ROC curves with AUC."""
    n_classes = y_prob.shape[1] if y_prob.ndim > 1 else len(np.unique(y_true))

    fig, ax = plt.subplots(figsize=(7, 6))

    for i in range(n_classes):
        y_binary = (y_true == i).astype(int)
        if y_prob.ndim > 1:
            fpr, tpr, _ = roc_curve(y_binary, y_prob[:, i])
        else:
            fpr, tpr, _ = roc_curve(y_binary, (y_prob == i).astype(float))
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[i], lw=2,
                label=f'{CLASS_LABELS[i]} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves — {model_name}')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  ✓ ROC curves → {output_path}")


# ============================================================
# 3. Accuracy vs SNR
# ============================================================
def plot_accuracy_vs_snr(
    snr_results: Dict[float, Dict[str, float]],
    output_path: str,
):
    """
    Plot accuracy vs SNR for multiple models.

    Args:
        snr_results: {snr_db: {model_name: accuracy}}
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    snr_values = sorted(snr_results.keys())
    model_names = list(next(iter(snr_results.values())).keys())

    markers = ['o', 's', '^', 'D', 'v', 'P', '*']
    for i, model_name in enumerate(model_names):
        accs = [snr_results[snr].get(model_name, 0) for snr in snr_values]
        ax.plot(snr_values, accs, marker=markers[i % len(markers)],
                lw=2, markersize=8, label=model_name)

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Classification Accuracy')
    ax.set_title('Classification Accuracy vs Signal-to-Noise Ratio')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  ✓ Accuracy vs SNR → {output_path}")


# ============================================================
# 4. t-SNE Visualization
# ============================================================
def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    output_path: str,
    perplexity: int = 30,
    seed: int = 42,
):
    """Plot t-SNE of learned feature embeddings."""
    print(f"  Computing t-SNE (perplexity={perplexity})...", end=" ", flush=True)

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                n_iter=1000, learning_rate='auto', init='pca')
    coords = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 7))
    for i in range(len(CLASS_LABELS)):
        mask = labels == i
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=COLORS[i], label=CLASS_LABELS[i],
                   alpha=0.6, s=15, edgecolors='none')

    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title(f't-SNE Feature Visualization — {model_name}')
    ax.legend(markerscale=3)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"done → {output_path}")


# ============================================================
# 5. Training Curves
# ============================================================
def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    model_name: str,
    output_path: str,
):
    """Plot loss and accuracy vs epoch."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs = range(1, len(train_losses) + 1)

    # Loss
    ax1.plot(epochs, train_losses, 'b-', lw=2, label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', lw=2, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training & Validation Loss — {model_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, train_accs, 'b-', lw=2, label='Train Acc')
    ax2.plot(epochs, val_accs, 'r-', lw=2, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Training & Validation Accuracy — {model_name}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  ✓ Training curves → {output_path}")


# ============================================================
# 6. Spectrogram Gallery
# ============================================================
def plot_spectrogram_gallery(
    data_dir: str,
    output_path: str,
    n_examples: int = 3,
):
    """Plot example spectrograms for each class."""
    dataset = IQDataset(data_dir, window_size=WINDOW_SIZE, max_windows_per_class=10)

    fig, axes = plt.subplots(len(CLASS_LABELS), n_examples,
                              figsize=(4 * n_examples, 3.5 * len(CLASS_LABELS)))

    for cls_idx, cls_name in enumerate(CLASS_LABELS):
        cls_mask = dataset.labels == cls_idx
        cls_windows = dataset.windows[cls_mask][:n_examples]

        for ex_idx in range(min(n_examples, len(cls_windows))):
            ax = axes[cls_idx, ex_idx] if len(CLASS_LABELS) > 1 else axes[ex_idx]
            spec = compute_spectrogram(cls_windows[ex_idx])
            ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
            if ex_idx == 0:
                ax.set_ylabel(cls_name, fontsize=12, fontweight='bold')
            if cls_idx == 0:
                ax.set_title(f'Example {ex_idx+1}')
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle('STFT Spectrogram Gallery — All Classes', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  ✓ Spectrogram gallery → {output_path}")


# ============================================================
# 7. Spike Visualization
# ============================================================
def plot_spike_overlay(
    data_dir: str,
    output_path: str,
):
    """Plot IQ signal with spike detection overlay for normal vs attack."""
    dataset = IQDataset(data_dir, window_size=1024, max_windows_per_class=1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    for cls_idx, (cls_name, ax) in enumerate(
        zip(['Clear', 'Human'], axes)
    ):
        cls_mask = dataset.labels == cls_idx
        if not np.any(cls_mask):
            continue
        window = dataset.windows[cls_mask][0]  # [1024, 2]
        amplitude = np.sqrt(window[:, 0]**2 + window[:, 1]**2)
        _, peaks, density = detect_spikes(window)

        ax.plot(amplitude, 'b-', alpha=0.7, lw=0.8, label='|IQ| Amplitude')
        # Overlay spike detections
        spike_indices = np.where(peaks > 0)[0]
        if len(spike_indices) > 0:
            ax.scatter(spike_indices, amplitude[spike_indices],
                       c='red', s=30, zorder=5, label=f'Spikes ({len(spike_indices)})')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'{cls_name} — Spike Density: {density:.4f}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Sample Index')
    fig.suptitle('Spike Detection: Normal vs Intrusion', fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  ✓ Spike overlay → {output_path}")


# ============================================================
# 8. Ablation Comparison Bar Chart
# ============================================================
def plot_ablation_comparison(
    results_path: str,
    output_path: str,
):
    """Plot bar chart comparing all model accuracies."""
    with open(results_path, 'r') as f:
        all_results = json.load(f)

    models = []
    accs = []
    stds = []
    colors_list = []

    color_map = {
        'spike_baseline': '#9E9E9E',
        'svm_baseline': '#795548',
        'mlp_baseline': '#FF9800',
        'cnn1d_iq': '#2196F3',
        'cnn2d_spec': '#4CAF50',
        'dual_branch_fusion': '#F44336',
        'dual_branch_lite': '#E91E63',
    }

    for name, res in all_results.items():
        if 'error' in res:
            continue
        models.append(name.replace('_', '\n'))
        accs.append(res.get('mean_accuracy', 0))
        stds.append(res.get('std_accuracy', 0))
        colors_list.append(color_map.get(name, '#666666'))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(models)), accs, yerr=stds, capsize=5,
                  color=colors_list, edgecolor='white', linewidth=1.5,
                  alpha=0.9)

    # Add value labels on bars
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{acc:.2%}', ha='center', va='bottom', fontsize=10,
                fontweight='bold')

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel('Classification Accuracy')
    ax.set_title('Ablation Study — Model Comparison')
    ax.set_ylim([0, 1.15])
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=accs[-1] if accs else 0, color='red', linestyle='--',
               alpha=0.3, label='Our Model')

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  ✓ Ablation comparison → {output_path}")


# ============================================================
# 9. Edge Inference Benchmark Table
# ============================================================
def plot_inference_benchmark(
    benchmark_results: Dict,
    output_path: str,
):
    """Plot inference time comparison as horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(9, 5))

    models = list(benchmark_results.keys())
    cpu_times = [benchmark_results[m].get('cpu_ms', 0) for m in models]
    params = [benchmark_results[m].get('params', 0) for m in models]

    y_pos = range(len(models))
    bars = ax.barh(y_pos, cpu_times, color=COLORS[:len(models)] + ['#9C27B0'] * 10,
                   edgecolor='white', height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([m.replace('_', ' ').title() for m in models])
    ax.set_xlabel('Inference Time (ms)')
    ax.set_title('Edge Device Inference Benchmark (CPU, batch=1)')
    ax.grid(axis='x', alpha=0.3)

    # Add labels
    for bar, t, p in zip(bars, cpu_times, params):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{t:.2f}ms | {p:,} params', va='center', fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  ✓ Inference benchmark → {output_path}")


# ============================================================
# Generate All Figures from Saved Results
# ============================================================
def generate_all_figures(
    results_dir: str = "results/models",
    data_dir: str = "dataset/simulated",
    output_dir: str = "results/figures",
):
    """Generate all paper figures from saved training results."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Generating Paper Figures")
    print(f"{'='*60}\n")

    # Load ablation results if available
    ablation_path = os.path.join(results_dir, "ablation_results.json")
    if os.path.exists(ablation_path):
        with open(ablation_path, 'r') as f:
            all_results = json.load(f)

        # Generate per-model figures
        for model_name, res in all_results.items():
            if 'error' in res or 'fold_results' not in res:
                continue

            # Use last fold results for confusion matrix
            last_fold = res['fold_results'][-1]
            y_true = np.array(last_fold['final_labels'])
            y_pred = np.array(last_fold['final_preds'])

            # Confusion matrix
            plot_confusion_matrix(
                y_true, y_pred, model_name,
                os.path.join(output_dir, f"cm_{model_name}.png"),
            )

            # Training curves (from last fold)
            plot_training_curves(
                last_fold['train_losses'], last_fold['val_losses'],
                last_fold['train_accs'], last_fold['val_accs'],
                model_name,
                os.path.join(output_dir, f"curves_{model_name}.png"),
            )

        # Ablation comparison
        plot_ablation_comparison(
            ablation_path,
            os.path.join(output_dir, "ablation_comparison.png"),
        )

    # Spectrogram gallery (always generate from data)
    if os.path.exists(data_dir):
        plot_spectrogram_gallery(
            data_dir,
            os.path.join(output_dir, "spectrogram_gallery.png"),
        )
        plot_spike_overlay(
            data_dir,
            os.path.join(output_dir, "spike_overlay.png"),
        )

    print(f"\n✅ All figures saved to {output_dir}/")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Generate evaluation figures")
    parser.add_argument("--results_dir", type=str, default="results/models")
    parser.add_argument("--data_dir", type=str, default="dataset/simulated")
    parser.add_argument("--output_dir", type=str, default="results/figures")
    args = parser.parse_args()

    generate_all_figures(args.results_dir, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()

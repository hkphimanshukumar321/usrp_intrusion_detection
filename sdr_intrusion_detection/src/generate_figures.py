#!/usr/bin/env python3
"""
generate_figures.py — IEEE Journal-Quality Figure Generator
============================================================
Generates publication-ready figures from SDR intrusion-detection
ablation study results.

Outputs saved to:  results/figures/

Usage:
    python src/generate_figures.py
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────
RESULTS_DIR = "results/models"
FIGURES_DIR = "results/figures"
DPI = 300
FIG_FORMAT = "pdf"  # "pdf" for LaTeX, "png" for quick preview

CLASS_NAMES = {0: "Clear", 1: "Human", 2: "Animal", 3: "Drone"}
NUM_CLASSES = 4

# IEEE-friendly display names
MODEL_DISPLAY = {
    "spike_baseline":      "Spike Detect.",
    "svm_baseline":        "SVM (RBF)",
    "mlp_baseline":        "MLP",
    "cnn1d_iq":            "1D-CNN (IQ)",
    "cnn2d_spec":          "2D-CNN (Spec.)",
    "dual_branch_fusion":  "Dual-Branch Fusion",
    "dual_branch_lite":    "Dual-Branch Lite",
}

# Curated color palette — distinguishable in print & screen
MODEL_COLORS = {
    "spike_baseline":      "#9e9e9e",
    "svm_baseline":        "#8d6e63",
    "mlp_baseline":        "#5c6bc0",
    "cnn1d_iq":            "#26a69a",
    "cnn2d_spec":          "#ef5350",
    "dual_branch_fusion":  "#1565c0",
    "dual_branch_lite":    "#ff8f00",
}

# Marker styles for line plots
MODEL_MARKERS = {
    "spike_baseline":      "v",
    "svm_baseline":        "^",
    "mlp_baseline":        "s",
    "cnn1d_iq":            "o",
    "cnn2d_spec":          "D",
    "dual_branch_fusion":  "P",
    "dual_branch_lite":    "*",
}


# ─── IEEE Matplotlib Style ───────────────────────────────────
def set_ieee_style():
    """Set matplotlib rc params matching IEEE Transactions style."""
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif"],
        "font.size":          9,
        "axes.labelsize":     10,
        "axes.titlesize":     11,
        "legend.fontsize":    8,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "axes.linewidth":     0.8,
        "grid.linewidth":     0.4,
        "lines.linewidth":    1.2,
        "lines.markersize":   5,
        "figure.dpi":         DPI,
        "savefig.dpi":        DPI,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "legend.framealpha":  0.9,
        "legend.edgecolor":   "0.8",
    })

set_ieee_style()


# ─── Data Loading ────────────────────────────────────────────
def load_all_results():
    """Load individual result JSONs + ablation master file."""
    results = {}

    # Load ablation_results.json (master file)
    ablation_path = os.path.join(RESULTS_DIR, "ablation_results.json")
    if os.path.exists(ablation_path):
        with open(ablation_path) as f:
            results = json.load(f)

    # Also load individual model JSONs (may override / supplement ablation)
    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith("_results.json") and fname != "ablation_results.json":
            model_key = fname.replace("_results.json", "")
            with open(os.path.join(RESULTS_DIR, fname)) as f:
                data = json.load(f)
                # Prefer individual file if it has fold_results
                if "fold_results" in data or model_key not in results:
                    results[model_key] = data

    return results


def get_dl_models(results):
    """Return only deep-learning models that have fold_results."""
    return {k: v for k, v in results.items()
            if "fold_results" in v and isinstance(v["fold_results"], list)}


def savefig(fig, name):
    """Save figure in both PDF and PNG."""
    for fmt in ["pdf", "png"]:
        path = os.path.join(FIGURES_DIR, f"{name}.{fmt}")
        fig.savefig(path, format=fmt)
    plt.close(fig)
    print(f"  ✓ {name}.pdf/png")


# ─── Figure 1: Model Accuracy Comparison Bar Chart ──────────
def fig_accuracy_comparison(results):
    """Grouped bar chart — mean accuracy ± std for all models."""
    # Filter out errored models
    models = {k: v for k, v in results.items()
              if "mean_accuracy" in v and v["mean_accuracy"] > 0}

    # Sort by accuracy
    sorted_models = sorted(models.items(), key=lambda x: x[1]["mean_accuracy"])
    names = [MODEL_DISPLAY.get(k, k) for k, _ in sorted_models]
    accs = [v["mean_accuracy"] * 100 for _, v in sorted_models]
    stds = [v.get("std_accuracy", 0) * 100 for _, v in sorted_models]
    colors = [MODEL_COLORS.get(k, "#666") for k, _ in sorted_models]

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    bars = ax.barh(range(len(names)), accs, xerr=stds, height=0.6,
                   color=colors, edgecolor="white", linewidth=0.5,
                   capsize=3, error_kw={"linewidth": 0.8})

    # Value labels
    for i, (acc, std) in enumerate(zip(accs, stds)):
        ax.text(acc + std + 0.8, i, f"{acc:.1f}%",
                va="center", ha="left", fontsize=7, fontweight="bold")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Mean Accuracy (%)")
    ax.set_xlim(0, max(accs) + 12)
    ax.set_title("Model Classification Accuracy Comparison")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    savefig(fig, "fig01_accuracy_comparison")


# ─── Figure 2: Confusion Matrices (all DL models) ───────────
def fig_confusion_matrices(results):
    """Grid of normalized confusion matrices for best fold of each DL model."""
    dl_models = get_dl_models(results)
    if not dl_models:
        print("  ⚠ No DL models with fold_results, skipping confusion matrices")
        return

    n = len(dl_models)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.2 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    class_labels = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]

    for idx, (model_key, data) in enumerate(sorted(dl_models.items())):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]

        # Use best fold
        fold_results = data["fold_results"]
        best_fold = max(fold_results, key=lambda f: f["best_val_acc"])
        preds = np.array(best_fold["final_preds"])
        labels = np.array(best_fold["final_labels"])

        cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues",
                       vmin=0, vmax=1)

        # Annotate cells
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color=color, fontweight="bold")

        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(class_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(class_labels, fontsize=7)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
        ax.set_title(MODEL_DISPLAY.get(model_key, model_key), fontsize=9,
                     fontweight="bold")

    # Hide empty subplots
    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    fig.suptitle("Normalized Confusion Matrices (Best Fold)", fontsize=11,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig(fig, "fig02_confusion_matrices")


# ─── Figure 3: Training & Validation Curves ─────────────────
def fig_training_curves(results):
    """Training loss and validation accuracy learning curves."""
    dl_models = get_dl_models(results)
    if not dl_models:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    for model_key, data in sorted(dl_models.items()):
        fold_results = data["fold_results"]
        best_fold = max(fold_results, key=lambda f: f["best_val_acc"])
        epochs = range(1, len(best_fold["train_losses"]) + 1)

        label = MODEL_DISPLAY.get(model_key, model_key)
        color = MODEL_COLORS.get(model_key, "#666")
        marker = MODEL_MARKERS.get(model_key, "o")

        # Plot every 5th marker to avoid clutter
        markevery = max(1, len(best_fold["train_losses"]) // 8)

        ax1.plot(epochs, best_fold["train_losses"], color=color,
                 label=label, marker=marker, markevery=markevery,
                 markersize=4)
        ax2.plot(epochs, [a * 100 for a in best_fold["val_accs"]],
                 color=color, label=label, marker=marker,
                 markevery=markevery, markersize=4)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("(a) Training Loss Convergence")
    ax1.legend(loc="upper right", fontsize=6, ncol=1)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy (%)")
    ax2.set_title("(b) Validation Accuracy")
    ax2.legend(loc="lower right", fontsize=6, ncol=1)

    fig.tight_layout()
    savefig(fig, "fig03_training_curves")


# ─── Figure 4: Per-Class F1-Score Heatmap ────────────────────
def fig_f1_heatmap(results):
    """Heatmap of per-class F1-score for each model."""
    dl_models = get_dl_models(results)
    if not dl_models:
        return

    class_labels = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
    model_names = []
    f1_matrix = []

    for model_key in sorted(dl_models.keys()):
        data = dl_models[model_key]
        best_fold = max(data["fold_results"], key=lambda f: f["best_val_acc"])
        preds = np.array(best_fold["final_preds"])
        labels = np.array(best_fold["final_labels"])

        _, _, f1, _ = precision_recall_fscore_support(
            labels, preds, labels=list(range(NUM_CLASSES)), zero_division=0
        )
        f1_matrix.append(f1)
        model_names.append(MODEL_DISPLAY.get(model_key, model_key))

    f1_matrix = np.array(f1_matrix)

    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    im = ax.imshow(f1_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    for i in range(len(model_names)):
        for j in range(NUM_CLASSES):
            color = "white" if f1_matrix[i, j] > 0.6 else "black"
            ax.text(j, i, f"{f1_matrix[i, j]:.3f}",
                    ha="center", va="center", fontsize=7,
                    color=color, fontweight="bold")

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(class_labels, fontsize=8)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=7)
    ax.set_xlabel("Target Class")
    ax.set_title("Per-Class F1-Score Comparison")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("F1-Score", fontsize=8)

    fig.tight_layout()
    savefig(fig, "fig04_f1_heatmap")


# ─── Figure 5: Radar Chart — Multi-Metric Comparison ────────
def fig_radar_chart(results):
    """Radar chart comparing precision, recall, F1, accuracy across models."""
    dl_models = get_dl_models(results)
    if not dl_models:
        return

    metrics_labels = ["Accuracy", "Precision\n(Macro)", "Recall\n(Macro)",
                      "F1-Score\n(Macro)", "Weighted\nF1"]
    num_metrics = len(metrics_labels)

    fig, ax = plt.subplots(figsize=(4.5, 4.5),
                           subplot_kw=dict(projection="polar"))

    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    for model_key in sorted(dl_models.keys()):
        data = dl_models[model_key]
        best_fold = max(data["fold_results"], key=lambda f: f["best_val_acc"])
        preds = np.array(best_fold["final_preds"])
        labels = np.array(best_fold["final_labels"])

        acc = np.mean(preds == labels)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0)
        _, _, f1_weighted, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0)

        values = [acc, p_macro, r_macro, f1_macro, f1_weighted]
        values += values[:1]  # Close

        color = MODEL_COLORS.get(model_key, "#666")
        label = MODEL_DISPLAY.get(model_key, model_key)
        ax.plot(angles, values, color=color, linewidth=1.5, label=label)
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_labels, fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=6)
    ax.set_title("Multi-Metric Radar Comparison", y=1.08, fontsize=11,
                 fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=7)

    fig.tight_layout()
    savefig(fig, "fig05_radar_chart")


# ─── Figure 6: Pareto Front — Accuracy vs Model Complexity ──
def fig_pareto_accuracy_vs_params(results):
    """Pareto trade-off: accuracy vs number of parameters."""
    models_with_params = {}
    for k, v in results.items():
        if "mean_accuracy" not in v or v["mean_accuracy"] <= 0:
            continue
        np_ = v.get("num_params")
        if np_ is None or isinstance(np_, str):
            continue
        if np_ == 0:
            np_ = 1  # log-safe for spike baseline
        models_with_params[k] = v

    if not models_with_params:
        print("  ⚠ No models with params data, skipping Pareto")
        return

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    xs, ys, labels_list, colors_list = [], [], [], []
    for k, v in sorted(models_with_params.items()):
        np_ = v["num_params"] if v["num_params"] > 0 else 1
        acc = v["mean_accuracy"] * 100
        xs.append(np_)
        ys.append(acc)
        labels_list.append(MODEL_DISPLAY.get(k, k))
        colors_list.append(MODEL_COLORS.get(k, "#666"))

    for i, (x, y, lab, col) in enumerate(zip(xs, ys, labels_list, colors_list)):
        marker = list(MODEL_MARKERS.values())[i % len(MODEL_MARKERS)]
        ax.scatter(x, y, c=col, s=100, marker=marker, zorder=5,
                   edgecolors="white", linewidth=0.8)
        # Offset text to avoid overlap
        offset_x = 1.3
        offset_y = 1.2
        ax.annotate(lab, (x, y), textcoords="offset points",
                    xytext=(8, 5 if i % 2 == 0 else -12),
                    fontsize=6.5, color=col, fontweight="bold")

    # Draw Pareto front line
    points = sorted(zip(xs, ys), key=lambda p: p[0])
    pareto_x, pareto_y = [points[0][0]], [points[0][1]]
    best_y = points[0][1]
    for px, py in points[1:]:
        if py >= best_y:
            pareto_x.append(px)
            pareto_y.append(py)
            best_y = py
    ax.plot(pareto_x, pareto_y, "k--", linewidth=0.8, alpha=0.5,
            label="Pareto Front")

    ax.set_xscale("log")
    ax.set_xlabel("Number of Parameters (log scale)")
    ax.set_ylabel("Mean Accuracy (%)")
    ax.set_title("Accuracy vs. Model Complexity Trade-off")
    ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    savefig(fig, "fig06_pareto_accuracy_params")


# ─── Figure 7: K-Fold Cross-Validation Box Plot ─────────────
def fig_kfold_boxplot(results):
    """Box plot showing fold-level accuracy distribution per model."""
    models_with_folds = {k: v for k, v in results.items()
                         if "fold_accuracies" in v and len(v["fold_accuracies"]) > 1
                         and v.get("mean_accuracy", 0) > 0}

    if not models_with_folds:
        print("  ⚠ No fold-level data, skipping box plot")
        return

    sorted_models = sorted(models_with_folds.items(),
                           key=lambda x: x[1]["mean_accuracy"])
    names = [MODEL_DISPLAY.get(k, k) for k, _ in sorted_models]
    fold_data = [np.array(v["fold_accuracies"]) * 100 for _, v in sorted_models]
    colors = [MODEL_COLORS.get(k, "#666") for k, _ in sorted_models]

    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    bp = ax.boxplot(fold_data, labels=names, patch_artist=True,
                    widths=0.5, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="black",
                                   markeredgecolor="black", markersize=4),
                    medianprops=dict(color="black", linewidth=1.2),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)

    ax.set_ylabel("Fold Accuracy (%)")
    ax.set_title("K-Fold Cross-Validation Accuracy Distribution")
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    savefig(fig, "fig07_kfold_boxplot")


# ─── Figure 8: Per-Class Precision & Recall Grouped Bars ────
def fig_precision_recall_bars(results):
    """Grouped bar chart of per-class precision and recall."""
    dl_models = get_dl_models(results)
    if not dl_models:
        return

    class_labels = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
    n_models = len(dl_models)
    n_classes = NUM_CLASSES

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))

    x = np.arange(n_classes)
    width = 0.8 / n_models

    for m_idx, (model_key, data) in enumerate(sorted(dl_models.items())):
        best_fold = max(data["fold_results"], key=lambda f: f["best_val_acc"])
        preds = np.array(best_fold["final_preds"])
        labels = np.array(best_fold["final_labels"])

        prec, rec, _, _ = precision_recall_fscore_support(
            labels, preds, labels=list(range(NUM_CLASSES)), zero_division=0)

        color = MODEL_COLORS.get(model_key, "#666")
        label = MODEL_DISPLAY.get(model_key, model_key)
        offset = (m_idx - n_models / 2 + 0.5) * width

        ax1.bar(x + offset, prec, width, color=color, label=label,
                edgecolor="white", linewidth=0.3)
        ax2.bar(x + offset, rec, width, color=color, label=label,
                edgecolor="white", linewidth=0.3)

    for ax, title in [(ax1, "(a) Precision"), (ax2, "(b) Recall")]:
        ax.set_xticks(x)
        ax.set_xticklabels(class_labels, fontsize=8)
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=5.5, loc="lower right", ncol=1)

    fig.tight_layout()
    savefig(fig, "fig08_precision_recall_bars")


# ─── Figure 9: ROC Curves (One-vs-Rest) ─────────────────────
def fig_roc_curves(results):
    """ROC curves (One-vs-Rest macro) for DL models."""
    dl_models = get_dl_models(results)
    if not dl_models:
        return

    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(7.0, 2.8))

    class_labels = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]

    for class_idx in range(NUM_CLASSES):
        ax = axes[class_idx]
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.6, alpha=0.4)

        for model_key, data in sorted(dl_models.items()):
            best_fold = max(data["fold_results"], key=lambda f: f["best_val_acc"])
            preds = np.array(best_fold["final_preds"])
            labels = np.array(best_fold["final_labels"])

            # Binarize for OvR
            y_true_bin = (labels == class_idx).astype(int)
            y_pred_bin = (preds == class_idx).astype(int)

            # Simple ROC from hard predictions (2 points + endpoints)
            fpr, tpr, _ = roc_curve(y_true_bin, y_pred_bin)
            roc_auc = auc(fpr, tpr)

            color = MODEL_COLORS.get(model_key, "#666")
            label = f"{MODEL_DISPLAY.get(model_key, model_key)} ({roc_auc:.2f})"
            ax.plot(fpr, tpr, color=color, linewidth=1.2, label=label)

        ax.set_title(f"{class_labels[class_idx]}", fontsize=9, fontweight="bold")
        ax.set_xlabel("FPR", fontsize=7)
        if class_idx == 0:
            ax.set_ylabel("TPR", fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=4.5, loc="lower right")
        ax.set_aspect("equal")

    fig.suptitle("One-vs-Rest ROC Curves", fontsize=11, fontweight="bold", y=1.03)
    fig.tight_layout()
    savefig(fig, "fig09_roc_curves")


# ─── Figure 10: Accuracy vs Epoch (all folds) ───────────────
def fig_fold_learning_curves(results):
    """Per-fold learning curves with mean ± std band for best model."""
    dl_models = get_dl_models(results)
    if not dl_models:
        return

    # Find best model
    best_key = max(dl_models.keys(),
                   key=lambda k: dl_models[k]["mean_accuracy"])
    data = dl_models[best_key]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    color = MODEL_COLORS.get(best_key, "#1565c0")

    all_train_losses = []
    all_val_accs = []

    for fold in data["fold_results"]:
        epochs = range(1, len(fold["train_losses"]) + 1)
        ax1.plot(epochs, fold["train_losses"], color=color, alpha=0.25,
                 linewidth=0.8)
        ax2.plot(epochs, [a * 100 for a in fold["val_accs"]], color=color,
                 alpha=0.25, linewidth=0.8)
        all_train_losses.append(fold["train_losses"])
        all_val_accs.append([a * 100 for a in fold["val_accs"]])

    # Mean ± std
    mean_loss = np.mean(all_train_losses, axis=0)
    std_loss = np.std(all_train_losses, axis=0)
    mean_acc = np.mean(all_val_accs, axis=0)
    std_acc = np.std(all_val_accs, axis=0)
    epochs = range(1, len(mean_loss) + 1)

    ax1.plot(epochs, mean_loss, color=color, linewidth=2, label="Mean")
    ax1.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss,
                     color=color, alpha=0.15, label="±1 Std")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title(f"(a) Training Loss — {MODEL_DISPLAY.get(best_key, best_key)}")
    ax1.legend(fontsize=7)

    ax2.plot(epochs, mean_acc, color=color, linewidth=2, label="Mean")
    ax2.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc,
                     color=color, alpha=0.15, label="±1 Std")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy (%)")
    ax2.set_title(f"(b) Val Accuracy — {MODEL_DISPLAY.get(best_key, best_key)}")
    ax2.legend(fontsize=7)

    fig.tight_layout()
    savefig(fig, "fig10_best_model_fold_curves")


# ─── Figure 11: Summary Table as Figure ─────────────────────
def fig_summary_table(results):
    """Render publication-quality results table as a figure."""
    rows = []
    for k, v in sorted(results.items()):
        if "mean_accuracy" not in v or v["mean_accuracy"] <= 0:
            continue
        np_ = v.get("num_params", "—")
        if isinstance(np_, int):
            np_str = f"{np_:,}"
        else:
            np_str = str(np_)
        mean_acc = v["mean_accuracy"] * 100
        std_acc = v.get("std_accuracy", 0) * 100

        # Compute macro F1 if fold_results available
        f1_str = "—"
        if "fold_results" in v and isinstance(v["fold_results"], list):
            best_fold = max(v["fold_results"], key=lambda f: f["best_val_acc"])
            preds = np.array(best_fold["final_preds"])
            labels = np.array(best_fold["final_labels"])
            _, _, f1_macro, _ = precision_recall_fscore_support(
                labels, preds, average="macro", zero_division=0)
            f1_str = f"{f1_macro:.4f}"

        rows.append([
            MODEL_DISPLAY.get(k, k),
            np_str,
            f"{mean_acc:.2f} ± {std_acc:.2f}",
            f1_str,
        ])

    col_labels = ["Model", "Parameters", "Accuracy (%)", "F1 (Macro)"]

    fig, ax = plt.subplots(figsize=(5.5, 0.4 * len(rows) + 1.0))
    ax.axis("off")

    table = ax.table(cellText=rows, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#1565c0")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("white")

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            cell.set_edgecolor("#e0e0e0")
            if i % 2 == 0:
                cell.set_facecolor("#f5f5f5")

    ax.set_title("Ablation Study — Summary of Results",
                 fontsize=11, fontweight="bold", pad=10)

    fig.tight_layout()
    savefig(fig, "fig11_summary_table")


# ─── Figure 12: Detection Accuracy per SNR-like condition ────
def fig_class_accuracy_breakdown(results):
    """Stacked / grouped bar showing per-class accuracy for each model."""
    dl_models = get_dl_models(results)
    if not dl_models:
        return

    class_labels = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
    n_models = len(dl_models)

    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    x = np.arange(n_models)
    width = 0.8 / NUM_CLASSES
    class_colors = ["#26a69a", "#ef5350", "#5c6bc0", "#ff8f00"]

    model_names = sorted(dl_models.keys())

    for c_idx in range(NUM_CLASSES):
        accs = []
        for model_key in model_names:
            data = dl_models[model_key]
            best_fold = max(data["fold_results"], key=lambda f: f["best_val_acc"])
            preds = np.array(best_fold["final_preds"])
            labels = np.array(best_fold["final_labels"])
            mask = labels == c_idx
            class_acc = np.mean(preds[mask] == labels[mask]) * 100
            accs.append(class_acc)

        offset = (c_idx - NUM_CLASSES / 2 + 0.5) * width
        ax.bar(x + offset, accs, width, color=class_colors[c_idx],
               label=class_labels[c_idx], edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY.get(k, k) for k in model_names],
                       rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("Per-Class Accuracy (%)")
    ax.set_title("Per-Class Detection Accuracy Breakdown")
    ax.legend(title="Target Class", fontsize=7, title_fontsize=8,
              loc="upper left", ncol=2)
    ax.set_ylim(0, 105)

    fig.tight_layout()
    savefig(fig, "fig12_class_accuracy_breakdown")


# ─── Figure 13: Pareto — F1 vs Inference Complexity ──────────
def fig_pareto_f1_vs_params(results):
    """Bubble chart: F1 vs params, bubble size = accuracy."""
    dl_models = get_dl_models(results)
    if not dl_models:
        return

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    for model_key, data in sorted(dl_models.items()):
        np_ = data.get("num_params", 0)
        if np_ == 0 or isinstance(np_, str):
            continue

        best_fold = max(data["fold_results"], key=lambda f: f["best_val_acc"])
        preds = np.array(best_fold["final_preds"])
        labels = np.array(best_fold["final_labels"])
        _, _, f1_macro, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0)

        acc = data["mean_accuracy"] * 100
        color = MODEL_COLORS.get(model_key, "#666")
        marker = MODEL_MARKERS.get(model_key, "o")
        label = MODEL_DISPLAY.get(model_key, model_key)

        ax.scatter(np_, f1_macro, s=acc * 3, c=color, marker=marker,
                   edgecolors="black", linewidth=0.5, zorder=5, alpha=0.85)
        ax.annotate(label, (np_, f1_macro),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=6, color=color, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("Number of Parameters (log scale)")
    ax.set_ylabel("Macro F1-Score")
    ax.set_title("F1-Score vs. Model Complexity (bubble size ∝ accuracy)")

    fig.tight_layout()
    savefig(fig, "fig13_pareto_f1_params")


# ─── Main ────────────────────────────────────────────────────
def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  IEEE Figure Generator — SDR Intrusion Detection")
    print(f"  Output: {FIGURES_DIR}/")
    print(f"{'='*60}\n")

    results = load_all_results()
    print(f"Loaded {len(results)} models: {list(results.keys())}\n")

    print("Generating figures...")
    fig_accuracy_comparison(results)
    fig_confusion_matrices(results)
    fig_training_curves(results)
    fig_f1_heatmap(results)
    fig_radar_chart(results)
    fig_pareto_accuracy_vs_params(results)
    fig_kfold_boxplot(results)
    fig_precision_recall_bars(results)
    fig_roc_curves(results)
    fig_fold_learning_curves(results)
    fig_summary_table(results)
    fig_class_accuracy_breakdown(results)
    fig_pareto_f1_vs_params(results)

    print(f"\n{'='*60}")
    n_files = len([f for f in os.listdir(FIGURES_DIR)
                   if f.startswith("fig")])
    print(f"  ✅ Done! {n_files} figure files saved to {FIGURES_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

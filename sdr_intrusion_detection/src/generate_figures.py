"""
generate_figures.py — IEEE-quality figures from ablation results
================================================================
Reads JSON logs from training and generates publication-ready plots.
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.config import (
    CUSTOM_MODEL_NAME, PLOT_COLORS, RADAR_CHART_TARGETS,
    FIGURES_DIR, RESULTS_DIR, LOGS_DIR,
)

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = PLOT_COLORS
CUSTOM_MODEL = CUSTOM_MODEL_NAME


def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"  [SKIP] {filepath} not found.")
        return None
    with open(filepath, 'r') as f:
        return json.load(f)


# ============================================================
# FIG 1: Training Curves (Loss & Accuracy, non-overlapping)
# ============================================================
def plot_training_curves(model_name=CUSTOM_MODEL):
    data = load_json(os.path.join(LOGS_DIR, f'history_{model_name}.json'))
    if not data:
        return

    history = data['history']
    epochs = [h['epoch'] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Left panel: Loss
    ax1.plot(epochs, [h['train_loss'] for h in history], 'o-', color='#e63946', label='Train Loss', markersize=3)
    ax1.plot(epochs, [h['val_loss'] for h in history], 's-', color='#457b9d', label='Val Loss', markersize=3)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Focal Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right panel: Accuracy
    ax2.plot(epochs, [h['train_acc'] for h in history], 'o-', color='#2a9d8f', label='Train Acc', markersize=3)
    ax2.plot(epochs, [h['val_acc'] for h in history], 's-', color='#e9c46a', label='Val Acc', markersize=3)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 105])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'{model_name}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'training_curves_{model_name}.png'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, f'training_curves_{model_name}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  [*] training_curves_{model_name}.png")


# ============================================================
# FIG 2: Radar Chart (Custom vs top baselines)
# ============================================================
def plot_radar_chart():
    data = load_json(os.path.join(RESULTS_DIR, 'ablation_backbone.json'))
    if not data:
        return

    targets = RADAR_CHART_TARGETS
    plot_data = {k: v for k, v in data.items() if k in targets}
    if len(plot_data) < 2:
        print("  [SKIP] Not enough models for radar chart.")
        return

    labels = ['Accuracy (%)', 'Param Efficiency', 'Speed', 'Memory']
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=300)

    for i, (name, m) in enumerate(plot_data.items()):
        acc = m['best_val_acc']
        param_eff = 100 * (1 / (m['profile']['Total_Parameters'] / 1e6 + 1))
        speed = 100 * (1 / (m['profile']['Inference_Time_ms'] / 10 + 1))
        mem = 100 * (1 / (m['profile']['Model_Size_MB'] / 10 + 1))

        vals = [acc, param_eff, speed, mem] + [acc]
        ax.plot(angles, vals, color=COLORS[i % len(COLORS)], linewidth=2, label=name)
        ax.fill(angles, vals, color=COLORS[i % len(COLORS)], alpha=0.08)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=11, fontweight='bold')
    plt.ylim(0, 100)
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)
    plt.title("Multi-Dimensional Backbone Comparison", fontsize=14, fontweight='bold', y=1.1)
    plt.savefig(os.path.join(FIGURES_DIR, 'radar_comparison.png'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'radar_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print("  [*] radar_comparison.png")


# ============================================================
# FIG 3: Pareto Scatter (Accuracy vs Params / Latency)
# ============================================================
def plot_pareto(metric_key, x_label, filename):
    data = load_json(os.path.join(RESULTS_DIR, 'ablation_backbone.json'))
    if not data:
        return

    names, x_vals, y_vals, colors_list = [], [], [], []
    for name, m in data.items():
        names.append(name)
        y_vals.append(m['best_val_acc'])
        val = m['profile'][metric_key]
        if metric_key == 'Total_Parameters':
            val = val / 1e6
        x_vals.append(val)
        colors_list.append('#e63946' if name == CUSTOM_MODEL else '#457b9d')

    plt.figure(figsize=(10, 7), dpi=300)
    plt.scatter(x_vals, y_vals, c=colors_list, s=120, alpha=0.8, edgecolors='k', linewidths=0.5)
    for i, txt in enumerate(names):
        plt.annotate(txt, (x_vals[i], y_vals[i]), xytext=(5, 5),
                     textcoords='offset points', fontsize=7, alpha=0.8)

    plt.title(f"Pareto: Accuracy vs {x_label}", fontsize=14, fontweight='bold')
    plt.xlabel(f"{x_label}" + (" (Millions)" if 'Param' in x_label else " (ms)"), fontsize=12)
    plt.ylabel("Validation Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.savefig(os.path.join(FIGURES_DIR, f'{filename}.png'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, f'{filename}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  [*] {filename}.png")


# ============================================================
# FIG 4: Optuna Hyperparameter Importance
# ============================================================
def plot_hparam_importance():
    data = load_json(os.path.join(RESULTS_DIR, 'ablation_hparam.json'))
    if not data:
        return

    trials = data['all_trials']
    if len(trials) < 3:
        return

    # Simple importance: variance of accuracy grouped by each param value
    param_names = list(trials[0]['params'].keys())
    importances = {}

    for p in param_names:
        groups = {}
        for t in trials:
            val = str(t['params'][p])
            groups.setdefault(val, []).append(t['value'] if t['value'] else 0)
        means = [np.mean(v) for v in groups.values()]
        importances[p] = np.std(means) if len(means) > 1 else 0

    # Sort
    sorted_params = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    names = [s[0] for s in sorted_params]
    vals = [s[1] for s in sorted_params]

    plt.figure(figsize=(10, 5), dpi=300)
    plt.barh(names, vals, color='#2a9d8f')
    plt.xlabel('Importance (Std of group means)', fontsize=12)
    plt.title('Hyperparameter Importance (Optuna)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'hparam_importance.png'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'hparam_importance.pdf'), bbox_inches='tight')
    plt.close()
    print("  [*] hparam_importance.png")


# ============================================================
# FIG 5: Backbone Accuracy Bar Chart
# ============================================================
def plot_accuracy_bars():
    data = load_json(os.path.join(RESULTS_DIR, 'ablation_backbone.json'))
    if not data:
        return

    sorted_models = sorted(data.items(), key=lambda x: x[1]['best_val_acc'], reverse=True)
    names = [s[0] for s in sorted_models]
    accs = [s[1]['best_val_acc'] for s in sorted_models]
    colors_list = ['#e63946' if n == CUSTOM_MODEL else '#457b9d' for n in names]

    plt.figure(figsize=(14, 6), dpi=300)
    bars = plt.bar(range(len(names)), accs, color=colors_list, edgecolor='k', linewidth=0.5)
    plt.xticks(range(len(names)), names, rotation=45, ha='right', fontsize=9)
    plt.ylabel('Validation Accuracy (%)', fontsize=12)
    plt.title('Backbone Ablation: Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.ylim([max(0, min(accs) - 10), 105])

    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                 f'{acc:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'accuracy_comparison.png'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'accuracy_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print("  [*] accuracy_comparison.png")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  GENERATING IEEE JOURNAL FIGURES")
    print("=" * 60)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    plot_training_curves(CUSTOM_MODEL)
    plot_accuracy_bars()
    plot_radar_chart()
    plot_pareto('Total_Parameters', 'Total Parameters', 'pareto_acc_vs_params')
    plot_pareto('Inference_Time_ms', 'Inference Latency', 'pareto_acc_vs_latency')
    plot_hparam_importance()

    print("\n  Confusion matrices are generated by evaluate.py")
    print("=" * 60)

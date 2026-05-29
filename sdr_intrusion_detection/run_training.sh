#!/bin/bash
# ============================================================
# run_training.sh — A100-Optimized Full Ablation Pipeline
# ============================================================
# Designed for Tesla A100 (32GB shared GPU memory).
# Phase 1 runs models in PARALLEL (default: 2 at a time).
# Phase 2 & 3 run sequentially, then generates figures.
#
# Usage:
#   chmod +x run_training.sh
#   nohup ./run_training.sh > training_full.log 2>&1 &
#
# Tune MAX_WORKERS based on your GPU memory:
#   1 = sequential (safest), 2 = moderate, 4 = aggressive
# ============================================================

set -e
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

MAX_WORKERS=${MAX_WORKERS:-2}   # Override with: MAX_WORKERS=4 ./run_training.sh

echo "=============================================="
echo "  SDR Intrusion Detection — Full Pipeline"
echo "  GPU: Tesla A100 (32GB)"
echo "  Parallel Workers: ${MAX_WORKERS}"
echo "  Started: $(date)"
echo "=============================================="

# -----------------------------------------------------------
# PHASE 1: Backbone Comparison (19 models, PARALLEL)
# Models run as independent subprocesses, ${MAX_WORKERS} at a time.
# Each subprocess loads its model separately on the GPU.
# -----------------------------------------------------------
echo ""
echo ">>> PHASE 1: Backbone Comparison (${MAX_WORKERS} parallel workers)"
python -m src.run_ablation --phase backbone --epochs 30 --batch_size 64 --max_workers ${MAX_WORKERS}

# -----------------------------------------------------------
# PHASE 2: Hyperparameter Sweep (Optuna, 20 trials)
# Only trains our custom model with different HP configs.
# Sequential — Optuna needs trial results to guide sampling.
# -----------------------------------------------------------
echo ""
echo ">>> PHASE 2: Hyperparameter Sweep"
python -m src.run_ablation --phase hparam --epochs 20 --n_trials 20

# -----------------------------------------------------------
# PHASE 3: Cross-Dataset Generalization
# Train on USRP -> Test on Radar (and vice versa)
# -----------------------------------------------------------
echo ""
echo ">>> PHASE 3: Cross-Dataset Generalization"
python -m src.run_ablation --phase cross --epochs 20

# -----------------------------------------------------------
# EDGE BENCHMARKING
# -----------------------------------------------------------
echo ""
echo ">>> Edge Inference Benchmarking"
python -m src.benchmark_edge --n_runs 200

# -----------------------------------------------------------
# FIGURE GENERATION
# -----------------------------------------------------------
echo ""
echo ">>> Generating IEEE Figures"
python -m src.generate_figures

echo ""
echo "=============================================="
echo "  ALL DONE!"
echo "  Finished: $(date)"
echo "  Results:  results/"
echo "  Figures:  results/figures/"
echo "  W&B:      https://wandb.ai"
echo "=============================================="

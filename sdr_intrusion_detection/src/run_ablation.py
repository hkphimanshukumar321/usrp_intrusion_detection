#!/usr/bin/env python3
"""
run_ablation.py — Complete ablation study runner.

Generates data, trains all models, benchmarks inference, and produces all figures.

Usage:
    python -m src.run_ablation                              # Full pipeline
    python -m src.run_ablation --skip_data                  # Skip data generation
    python -m src.run_ablation --quick --duration 2 --epochs 10  # Quick smoke test
"""

import argparse
import json
import os
import time
from datetime import timedelta

from src.sim_system import generate_dataset
from src.train import train_all_models
from src.benchmark_edge import benchmark_all_models
from src.evaluate import generate_all_figures


def run_full_pipeline(
    data_dir: str = "dataset/simulated",
    results_dir: str = "results/models",
    figures_dir: str = "results/figures",
    duration: float = 10.0,
    snr_db: float = 20.0,
    epochs: int = 50,
    batch_size: int = 128,
    skip_data: bool = False,
    skip_train: bool = False,
    skip_benchmark: bool = False,
    quick: bool = False,
):
    """
    Run the complete pipeline: generate → train → benchmark → evaluate.
    """
    start_time = time.time()

    if quick:
        duration = min(duration, 2.0)
        epochs = min(epochs, 10)
        print("\n⚡ QUICK MODE: Reduced duration & epochs for testing\n")

    # =========================================
    # Phase 1: Generate Data
    # =========================================
    if not skip_data:
        print("\n" + "=" * 60)
        print("  PHASE 1: Generating Simulated Dataset")
        print("=" * 60)

        generate_dataset(
            output_dir=data_dir,
            duration_sec=duration,
            snr_db=snr_db,
        )
    else:
        print("\n  ⏩ Skipping data generation (--skip_data)")

    # =========================================
    # Phase 2: Train All Models
    # =========================================
    if not skip_train:
        print("\n" + "=" * 60)
        print("  PHASE 2: Training All Models (Ablation Study)")
        print("=" * 60)

        config = {
            'epochs': epochs,
            'batch_size': batch_size,
            'patience': max(5, epochs // 5),
        }

        all_results = train_all_models(data_dir, results_dir, config)
    else:
        print("\n  ⏩ Skipping training (--skip_train)")

    # =========================================
    # Phase 3: Benchmark Inference
    # =========================================
    if not skip_benchmark:
        print("\n" + "=" * 60)
        print("  PHASE 3: Edge Inference Benchmarking")
        print("=" * 60)

        benchmark_results = benchmark_all_models(results_dir)
    else:
        print("\n  ⏩ Skipping benchmark (--skip_benchmark)")

    # =========================================
    # Phase 4: Generate Figures
    # =========================================
    print("\n" + "=" * 60)
    print("  PHASE 4: Generating Paper Figures")
    print("=" * 60)

    generate_all_figures(results_dir, data_dir, figures_dir)

    # =========================================
    # Summary
    # =========================================
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  ✅ PIPELINE COMPLETE")
    print(f"  Total time: {timedelta(seconds=int(elapsed))}")
    print(f"  Data:       {data_dir}/")
    print(f"  Models:     {results_dir}/")
    print(f"  Figures:    {figures_dir}/")
    print(f"{'='*60}\n")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Run complete ablation study pipeline"
    )
    parser.add_argument("--data_dir", type=str, default="dataset/simulated")
    parser.add_argument("--results_dir", type=str, default="results/models")
    parser.add_argument("--figures_dir", type=str, default="results/figures")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Data generation duration per class (seconds)")
    parser.add_argument("--snr_db", type=float, default=20.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--skip_data", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_benchmark", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (2s data, 10 epochs)")
    args = parser.parse_args()

    run_full_pipeline(**vars(args))


if __name__ == "__main__":
    main()

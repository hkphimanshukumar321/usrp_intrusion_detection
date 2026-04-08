#!/usr/bin/env python3
"""
benchmark_edge.py — Inference time profiling for edge device deployment.

Benchmarks:
  1. PyTorch CPU inference (simulates Raspberry Pi / laptop)
  2. PyTorch GPU inference (if available)
  3. ONNX Runtime inference (optimized for edge)
  4. Model size on disk (bytes)
  5. FLOPs estimation

Usage:
    python -m src.benchmark_edge --model dual_branch_fusion
    python -m src.benchmark_edge --all
"""

import argparse
import json
import os
import pickle
import time
import numpy as np
import torch
from typing import Dict

from src.model import (
    build_model, get_model_input_mode, MODEL_REGISTRY,
    WINDOW_SIZE, NUM_CLASSES, SPEC_FREQ_BINS, SPEC_TIME_BINS,
)
from src.data_loader import IQDataset
from src.feature_extraction import compute_statistical_features_batch


# ============================================================
# Inference Timer
# ============================================================
def benchmark_pytorch(
    model: torch.nn.Module,
    model_name: str,
    device: torch.device,
    n_runs: int = 100,
    warmup: int = 10,
) -> Dict:
    """
    Benchmark PyTorch inference time.

    Args:
        model: PyTorch model
        model_name: Name for logging
        device: CPU or CUDA device
        n_runs: Number of inference runs
        warmup: Warmup runs (discarded)

    Returns:
        Dict with timing statistics
    """
    model = model.to(device)
    model.eval()

    input_mode = get_model_input_mode(model_name)

    # Create dummy inputs
    iq = torch.randn(1, WINDOW_SIZE, 2).to(device)
    if input_mode in {"dual", "spectrogram"}:
        spec = torch.randn(1, 1, SPEC_FREQ_BINS, SPEC_TIME_BINS).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            if input_mode == "dual":
                _ = model(iq, spec)
            elif input_mode == "spectrogram":
                _ = model(spec)
            else:
                _ = model(iq)

    # Timed runs
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            if input_mode == "dual":
                _ = model(iq, spec)
            elif input_mode == "spectrogram":
                _ = model(spec)
            else:
                _ = model(iq)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)  # ms

    return {
        'device': str(device),
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'median_ms': np.median(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'p95_ms': np.percentile(times, 95),
        'throughput_hz': 1000.0 / np.mean(times),
    }


# ============================================================
# ONNX Export & Benchmark
# ============================================================
def export_and_benchmark_onnx(
    model: torch.nn.Module,
    model_name: str,
    output_dir: str = "results/models",
    n_runs: int = 100,
) -> Dict:
    """
    Export model to ONNX and benchmark with ONNX Runtime.

    Returns:
        Dict with timing stats and model size
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("  ⚠ onnx/onnxruntime not installed, skipping ONNX benchmark")
        return {'error': 'onnx not installed'}

    model.eval()
    input_mode = get_model_input_mode(model_name)

    iq = torch.randn(1, WINDOW_SIZE, 2)
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
    os.makedirs(output_dir, exist_ok=True)

    try:
        if input_mode == "dual":
            spec = torch.randn(1, 1, SPEC_FREQ_BINS, SPEC_TIME_BINS)
            torch.onnx.export(
                model, (iq, spec), onnx_path,
                input_names=['iq', 'spectrogram'],
                output_names=['logits'],
                dynamic_axes={'iq': {0: 'batch'}, 'spectrogram': {0: 'batch'}},
                opset_version=13,
            )
            ort_inputs = {
                'iq': iq.numpy(),
                'spectrogram': spec.numpy(),
            }
        elif input_mode == "spectrogram":
            spec = torch.randn(1, 1, SPEC_FREQ_BINS, SPEC_TIME_BINS)
            torch.onnx.export(
                model, spec, onnx_path,
                input_names=['spectrogram'],
                output_names=['logits'],
                opset_version=13,
            )
            ort_inputs = {'spectrogram': spec.numpy()}
        else:
            torch.onnx.export(
                model, iq, onnx_path,
                input_names=['iq'],
                output_names=['logits'],
                opset_version=13,
            )
            ort_inputs = {'iq': iq.numpy()}

        # Get model size
        model_size_bytes = os.path.getsize(onnx_path)

        # Benchmark ONNX Runtime
        sess = ort.InferenceSession(onnx_path)

        # Warmup
        for _ in range(10):
            sess.run(None, ort_inputs)

        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            sess.run(None, ort_inputs)
            times.append((time.perf_counter() - start) * 1000)

        return {
            'onnx_path': onnx_path,
            'model_size_kb': model_size_bytes / 1024,
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'median_ms': np.median(times),
            'throughput_hz': 1000.0 / np.mean(times),
        }
    except Exception as e:
        return {'error': str(e)}


def export_pytorch_state_dict(
    model: torch.nn.Module,
    model_name: str,
    output_dir: str = "results/models",
) -> Dict:
    """Save a PyTorch state_dict and report its size."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{model_name}_state_dict.pt")
    torch.save(model.state_dict(), path)
    return {
        'state_dict_path': path,
        'model_size_kb': os.path.getsize(path) / 1024,
    }


def load_benchmark_windows(
    data_dir: str = None,
    max_windows_per_class: int = 4,
) -> np.ndarray:
    """Load a small real sample set for baseline benchmarking when possible."""
    if data_dir and os.path.isdir(data_dir):
        try:
            ds = IQDataset(
                data_dir,
                window_size=WINDOW_SIZE,
                max_windows_per_class=max_windows_per_class,
            )
            return ds.windows.astype(np.float32)
        except Exception:
            pass

    rng = np.random.default_rng(42)
    return rng.standard_normal((max_windows_per_class * NUM_CLASSES, WINDOW_SIZE, 2)).astype(np.float32)


def benchmark_baseline_artifact(
    artifact_path: str,
    baseline_name: str,
    sample_windows: np.ndarray,
    n_runs: int = 100,
    warmup: int = 10,
) -> Dict:
    """Benchmark saved SVM/spike baseline artifacts on CPU."""
    with open(artifact_path, 'rb') as f:
        artifact = pickle.load(f)

    sample = sample_windows[:1]
    if baseline_name == "svm_baseline":
        features = compute_statistical_features_batch(sample)

        def infer():
            scaler = artifact['scaler']
            model = artifact['model']
            return model.predict(scaler.transform(features))

        num_params = "N/A (non-parametric)"
    elif baseline_name == "spike_baseline":
        clf = artifact

        def infer():
            return clf.predict(sample)

        num_params = 0
    else:
        raise ValueError(f"Unsupported baseline artifact: {baseline_name}")

    for _ in range(warmup):
        infer()

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        infer()
        times.append((time.perf_counter() - start) * 1000)

    return {
        'params': num_params,
        'cpu_ms': np.mean(times),
        'cpu_std': np.std(times),
        'cpu_throughput': 1000.0 / np.mean(times),
        'gpu_ms': None,
        'gpu_throughput': None,
        'onnx_ms': None,
        'onnx_size_kb': os.path.getsize(artifact_path) / 1024,
        'pytorch_size_kb': os.path.getsize(artifact_path) / 1024,
        'artifact_path': artifact_path,
    }


# ============================================================
# Full Benchmark Suite
# ============================================================
def benchmark_all_models(
    output_dir: str = "results/models",
    model_names=None,
    data_dir: str = None,
) -> Dict:
    """Benchmark all models and generate comparison table."""
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    selected_models = model_names or list(MODEL_REGISTRY.keys())
    sample_windows = None

    print(f"\n{'='*60}")
    print(f"  Edge Device Inference Benchmark")
    print(f"{'='*60}\n")

    for name in selected_models:
        if name in {"svm_baseline", "spike_baseline"}:
            if sample_windows is None:
                sample_windows = load_benchmark_windows(data_dir)
            artifact_name = f"{name}_artifact.pkl"
            artifact_path = os.path.join(output_dir, artifact_name)
            if not os.path.exists(artifact_path):
                print(f"  Skipping {name}: artifact not found at {artifact_path}")
                continue
            print(f"\n  Benchmarking: {name}")
            results[name] = benchmark_baseline_artifact(
                artifact_path,
                name,
                sample_windows,
            )
            continue

        if name not in MODEL_REGISTRY:
            continue
        print(f"\n  Benchmarking: {name}")
        model = build_model(name)
        n_params = model.count_params()

        # CPU benchmark
        cpu_results = benchmark_pytorch(model, name, torch.device('cpu'))

        # GPU benchmark (if available)
        gpu_results = {}
        if torch.cuda.is_available():
            gpu_results = benchmark_pytorch(model, name, torch.device('cuda'))

        # PyTorch artifact size
        pytorch_results = export_pytorch_state_dict(model, name, output_dir)

        # ONNX benchmark
        onnx_results = export_and_benchmark_onnx(model, name, output_dir)

        results[name] = {
            'params': n_params,
            'cpu_ms': cpu_results['mean_ms'],
            'cpu_std': cpu_results['std_ms'],
            'cpu_throughput': cpu_results['throughput_hz'],
            'gpu_ms': gpu_results.get('mean_ms', None),
            'gpu_throughput': gpu_results.get('throughput_hz', None),
            'onnx_ms': onnx_results.get('mean_ms', None),
            'onnx_size_kb': onnx_results.get('model_size_kb', None),
            'pytorch_size_kb': pytorch_results.get('model_size_kb', None),
        }

    # Print table
    print(f"\n{'='*90}")
    print(f"  {'Model':<25} {'Params':>10} {'CPU (ms)':>10} {'GPU (ms)':>10} "
          f"{'ONNX (ms)':>10} {'Size (KB)':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for name, res in results.items():
        params = res['params']
        params_str = f"{params:,}" if isinstance(params, int) else str(params)
        gpu_str = f"{res['gpu_ms']:.2f}" if res['gpu_ms'] else "N/A"
        onnx_str = f"{res['onnx_ms']:.2f}" if res['onnx_ms'] else "N/A"
        size_kb = res.get('onnx_size_kb') or res.get('pytorch_size_kb')
        size_str = f"{size_kb:.0f}" if size_kb else "N/A"
        print(f"  {name:<25} {params_str:>10} {res['cpu_ms']:>10.2f} "
              f"{gpu_str:>10} {onnx_str:>10} {size_str:>10}")

    print(f"{'='*90}\n")

    # Save
    path = os.path.join(output_dir, "benchmark_results.json")
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved to: {path}")

    return results


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Edge inference benchmarking")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model to benchmark (or --all)")
    parser.add_argument("--all", action="store_true",
                        help="Benchmark all models")
    parser.add_argument("--output_dir", type=str, default="results/models")
    parser.add_argument("--data_dir", type=str, default="dataset/simulated")
    parser.add_argument("--n_runs", type=int, default=100)
    args = parser.parse_args()

    if args.all or args.model is None:
        benchmark_all_models(args.output_dir, data_dir=args.data_dir)
    else:
        if args.model in {"svm_baseline", "spike_baseline"}:
            results = benchmark_all_models(
                args.output_dir,
                model_names=[args.model],
                data_dir=args.data_dir,
            )
            print(json.dumps(results.get(args.model, {}), indent=2, default=str))
            return

        model = build_model(args.model)
        print(f"Params: {model.count_params():,}")
        cpu_res = benchmark_pytorch(model, args.model, torch.device('cpu'),
                                     n_runs=args.n_runs)
        print(f"CPU: {cpu_res['mean_ms']:.2f} ± {cpu_res['std_ms']:.2f} ms")
        print(f"Throughput: {cpu_res['throughput_hz']:.0f} inferences/sec")


if __name__ == "__main__":
    main()

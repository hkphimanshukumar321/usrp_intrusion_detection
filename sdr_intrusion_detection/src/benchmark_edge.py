"""
benchmark_edge.py — Inference latency profiling for edge deployment
===================================================================
Benchmarks all models on both CPU and GPU (A100-aware with CUDA sync).
"""
import argparse
import time
import torch
from src.model import get_model, TIMM_MODEL_MAP

CUSTOM_MODEL = 'SDR_Custom_CoordASPP_Focal'


def benchmark_pytorch(model, device, n_runs=100):
    """Benchmark a single model on a given device."""
    model.to(device).eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    return {
        'mean_ms': round(sum(times) / len(times), 2),
        'min_ms': round(min(times), 2),
        'max_ms': round(max(times), 2),
        'std_ms': round((sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5, 2)
    }


def run_benchmarks(n_runs=100):
    models = [CUSTOM_MODEL] + list(TIMM_MODEL_MAP.keys())

    devices = [torch.device('cpu')]
    if torch.cuda.is_available():
        devices.append(torch.device('cuda'))

    print("\n" + "=" * 80)
    print("  EDGE INFERENCE BENCHMARKING")
    print("=" * 80)

    results = {}
    for model_name in models:
        print(f"\n  [{model_name}]")
        try:
            model = get_model(model_name)
            total_params = sum(p.numel() for p in model.parameters())
            size_mb = round(total_params * 4 / (1024**2), 2)
            print(f"    Params: {total_params:,}  |  Size: {size_mb} MB")

            model_results = {"params": total_params, "size_mb": size_mb}
            for device in devices:
                res = benchmark_pytorch(model, device, n_runs)
                tag = device.type.upper()
                model_results[tag] = res
                print(f"    {tag}: {res['mean_ms']} ms (min={res['min_ms']}, max={res['max_ms']})")

            results[model_name] = model_results
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"    FAILED: {e}")

    # Save results
    import json, os
    os.makedirs('results', exist_ok=True)
    with open('results/edge_benchmark.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nBenchmark complete. Results -> results/edge_benchmark.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_runs", type=int, default=100)
    args = parser.parse_args()
    run_benchmarks(args.n_runs)

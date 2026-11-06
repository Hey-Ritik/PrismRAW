"""Isolated CPU pipeline latency and sampled process-memory benchmarks."""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import numpy as np
import psutil
import torch

from smartphone_isp.pipelines.classical import process
from smartphone_isp.pipelines.hybrid import load_model as load_hybrid, process_hybrid
from smartphone_isp.pipelines.learned import load_model as load_learned, process_learned


def worker(method, iterations):
    torch.set_num_threads(2)
    raw = np.load("data/processed/demo/003-noisy.npy", allow_pickle=False)[:256, :256]
    metadata = json.loads(Path("data/processed/demo/003.json").read_text())
    model = None
    path = None
    if method != "classical":
        path = Path("checkpoints") / {"learned": "final-learned.pt", "hybrid": "final-hybrid.pt", "optimized": "final-hybrid-fp16.pt"}[method]
        model, _ = (load_learned if method == "learned" else load_hybrid)(path)
    def run():
        if method == "classical":
            return process(raw, metadata)
        if method == "learned":
            return process_learned(raw, metadata, model, exposure=1.0)
        return process_hybrid(raw, metadata, model)
    for _ in range(3):
        run()
    process_info = psutil.Process()
    baseline = process_info.memory_info().rss
    peak = [baseline]
    stop = threading.Event()
    def sample():
        while not stop.wait(0.002):
            peak[0] = max(peak[0], process_info.memory_info().rss)
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    timings = []
    try:
        for _ in range(iterations):
            before = time.perf_counter()
            run()
            timings.append((time.perf_counter()-before)*1000)
            peak[0] = max(peak[0], process_info.memory_info().rss)
    finally:
        stop.set()
        sampler.join()
    return {"median_ms": float(np.median(timings)), "p95_ms": float(np.percentile(timings, 95)),
            "peak_process_rss_mib": peak[0]/2**20, "incremental_peak_rss_mib": (peak[0]-baseline)/2**20,
            "parameters": sum(p.numel() for p in model.parameters()) if model is not None else 0,
            "model_bytes": path.stat().st_size if path else 0, "iterations": iterations, "warmups": 3,
            "input_shape": list(raw.shape), "precision": str(next(model.parameters()).dtype) if model is not None else "float32"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("classical", "learned", "hybrid", "optimized"))
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if args.iterations < 3:
        raise ValueError("Use at least three timed iterations")
    if args.worker:
        print(json.dumps(worker(args.worker, args.iterations)))
        return
    results = {}
    for method in ("classical", "learned", "hybrid", "optimized"):
        result = subprocess.run([sys.executable, "-m", "smartphone_isp.evaluation.benchmark", "--worker", method, "--iterations", str(args.iterations)], capture_output=True, text=True, check=True)
        results[method] = json.loads(result.stdout.strip().splitlines()[-1])
        print(f"Benchmarked {method}", flush=True)
    report = {"hardware": {"platform": platform.platform(), "processor": platform.processor(), "logical_cpus": os.cpu_count(), "torch": str(torch.__version__), "device": "CPU", "torch_threads": 2},
              "scope": "Isolated process per method; complete array processing, excludes model loading, file I/O, network and metrics. Sampled RSS includes Python/PyTorch runtime, not model-only memory. Not a smartphone measurement.", "methods": results}
    Path("reports/benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Wrote reports/benchmark.json")


if __name__ == "__main__":
    main()

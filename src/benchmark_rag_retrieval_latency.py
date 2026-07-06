#!/usr/bin/env python
"""Benchmark SOMA RAG retrieval latency.

This measures the online `SOMAAgent.init_episode()` path: image/text embedding
plus memory retrieval with the current top-k settings used by the agent.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image

from soma_agent import SOMAAgent


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    idx = int(p * len(ordered)) - 1
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean_s": statistics.mean(values),
        "median_s": statistics.median(values),
        "p95_s": _percentile(values, 0.95),
        "p99_s": _percentile(values, 0.99),
        "min_s": min(values),
        "max_s": max(values),
        "mean_ms": statistics.mean(values) * 1000,
        "median_ms": statistics.median(values) * 1000,
        "p95_ms": _percentile(values, 0.95) * 1000,
        "p99_ms": _percentile(values, 0.99) * 1000,
        "min_ms": min(values) * 1000,
        "max_ms": max(values) * 1000,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--memory_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam3_url", default="http://127.0.0.1:5001")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--out", default="/mnt/disk1/shared_data/lzy/SOMA-local/src/outputs/latency_benchmarks/rag_retrieval_latency.json")
    args = parser.parse_args()

    image = np.array(Image.open(args.image).convert("RGB"))
    agent = SOMAAgent(
        {
            "device": args.device,
            "sam3_base_url": args.sam3_url,
            "memory_dir": args.memory_dir,
        }
    )

    last_ctx = None
    for _ in range(args.warmup):
        last_ctx = agent.init_episode(image, args.task)

    times: list[float] = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        last_ctx = agent.init_episode(image, args.task)
        times.append(time.perf_counter() - t0)

    result = {
        "benchmark": "RAG retrieval",
        "image": args.image,
        "task": args.task,
        "memory_dir": args.memory_dir,
        "device": args.device,
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "last_ctx_sizes": {k: len(v) for k, v in (last_ctx or {}).items()},
        "latency": _stats(times),
    }

    print(json.dumps(result, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("saved_to=", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

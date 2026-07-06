#!/usr/bin/env python
"""Benchmark VLM orchestration latency for non-visual SOMA tools.

Prompt-Refiner and Chaining-Step are dominated by the VLM planning call, so this
script measures `Qwen3VLAPIClient.orchestrate_perception` under prompts that
encourage each tool family.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image

from soma_vlm import Qwen3VLAPIClient


SCENARIOS = {
    "generic": {
        "task": "pick up the central black bowl and place it on the plate",
        "rag_context": "Past failures indicate visual ambiguity and distractor interference.",
    },
    "prompt_refiner": {
        "task": "Umm... can you grab that dark round thing in the middle, you know, the bowl, and put it on the plate?",
        "rag_context": "Past failures indicate linguistic ambiguity and noisy informal instructions. Prefer prompt refinement when visual input is clear.",
    },
    "chainstep": {
        "task": "Pick up the cream cheese and place it in the basket, then pick up the milk and place it in the basket, then pick up the chocolate pudding and place it in the basket.",
        "rag_context": "Past failures indicate long-horizon compounding errors. Prefer task_decompose into short subtasks with control-flow reset.",
    },
}


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
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    image = np.array(Image.open(args.image).convert("RGB"))
    scenario = SCENARIOS[args.scenario]

    vlm = Qwen3VLAPIClient(
        api_key=os.environ.get("SOMA_VLM_API_KEY"),
        base_url=os.environ.get("SOMA_VLM_BASE_URL"),
        model_id=os.environ.get("SOMA_VLM_MODEL_ID"),
    )

    last_plan = None
    for _ in range(args.warmup):
        last_plan = vlm.orchestrate_perception(
            image=image,
            task_desc=scenario["task"],
            rag_context=scenario["rag_context"],
            rag_hints={},
        )

    times: list[float] = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        last_plan = vlm.orchestrate_perception(
            image=image,
            task_desc=scenario["task"],
            rag_context=scenario["rag_context"],
            rag_hints={},
        )
        times.append(time.perf_counter() - t0)

    result = {
        "benchmark": f"VLM orchestration: {args.scenario}",
        "scenario": args.scenario,
        "image": args.image,
        "task": scenario["task"],
        "rag_context": scenario["rag_context"],
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "model_id": vlm.model_id,
        "base_url": vlm.base_url,
        "latency": _stats(times),
        "last_plan": last_plan,
    }

    print(json.dumps(result, indent=2))
    out_path = args.out or f"/mnt/disk1/shared_data/lzy/SOMA-local/src/outputs/latency_benchmarks/vlm_{args.scenario}_latency.json"
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("saved_to=", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

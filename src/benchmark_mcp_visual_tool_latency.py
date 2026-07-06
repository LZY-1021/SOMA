#!/usr/bin/env python
"""Benchmark end-to-end MCP visual tool latency over HTTP.

This measures the full client-side MCP call path, including image serialization,
HTTP request/response, server-side VLM-assisted localization if enabled,
SAM3 mask extraction, image editing, and response decoding.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image

from soma_tools import MCPTools


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
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--operation", choices=["overlay", "remove"], required=True)
    parser.add_argument("--sam3_url", default="http://127.0.0.1:5001")
    parser.add_argument("--timeout_s", type=float, default=60.0)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    image = np.array(Image.open(args.image).convert("RGB"))
    tools = MCPTools(sam3_base_url=args.sam3_url, timeout_s=args.timeout_s)

    def call_tool() -> np.ndarray:
        if args.operation == "overlay":
            return tools.apply_visual_overlay(image, args.prompt, color=(0, 255, 0))
        if args.operation == "remove":
            return tools.remove_distractor(image, args.prompt)
        raise ValueError(args.operation)

    for _ in range(args.warmup):
        _ = call_tool()

    times: list[float] = []
    last_shape = None
    for _ in range(args.runs):
        t0 = time.perf_counter()
        out_img = call_tool()
        times.append(time.perf_counter() - t0)
        last_shape = list(out_img.shape)

    result = {
        "benchmark": f"MCP visual tool end-to-end: {args.operation}",
        "image": args.image,
        "prompt": args.prompt,
        "operation": args.operation,
        "sam3_url": args.sam3_url,
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "last_output_shape": last_shape,
        "latency": _stats(times),
    }

    print(json.dumps(result, indent=2))
    out_path = args.out or f"/mnt/disk1/shared_data/lzy/SOMA-local/src/outputs/latency_benchmarks/mcp_{args.operation}_e2e_latency.json"
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("saved_to=", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

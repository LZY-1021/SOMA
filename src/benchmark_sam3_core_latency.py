#!/usr/bin/env python
"""Benchmark local SAM3 mask extraction and visual post-processing latency.

This script intentionally bypasses the SOMA MCP HTTP path and disables the
VLM-assisted bbox localizer. It measures only the local SAM3 text-prompt mask
path plus lightweight post-processing used by /visual_overlay or
/remove_distractor.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image

import sam3_service


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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


def _overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = image.copy()
    overlay = np.zeros_like(out)
    overlay[mask == 1] = np.array(color, dtype=np.uint8)
    out = (out * (1 - alpha) + overlay * alpha).astype(np.uint8)
    out[mask == 0] = image[mask == 0]
    return out


def _remove(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((7, 7), np.uint8)
    mask_d = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(img_bgr, mask_d, 5, cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)


def _postprocess(operation: str, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if operation == "overlay":
        return _overlay(image, mask, color=(0, 255, 0), alpha=0.4)
    if operation == "remove":
        return _remove(image, mask)
    raise ValueError(f"Unsupported operation: {operation}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--operation", choices=["overlay", "remove"], default="overlay")
    parser.add_argument("--sam3_weight_path", default="/mnt/disk1/shared_data/lzy/models/sam/sam3.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    image = np.array(Image.open(args.image).convert("RGB"))

    ok = sam3_service.init_sam3_model(device=args.device, sam3_weight_path=args.sam3_weight_path)
    if not ok:
        raise RuntimeError("Failed to initialize SAM3")

    # Disable VLM-assisted bbox localization; benchmark pure SAM3 text-prompt mask path.
    sam3_service.vlm_client = None

    for _ in range(args.warmup):
        mask, _ = sam3_service._get_mask(image, args.prompt)
        _ = _postprocess(args.operation, image, mask)
        _sync_cuda()

    mask_times: list[float] = []
    overlay_times: list[float] = []
    total_times: list[float] = []
    last_score = 0.0
    last_mask_pixels = 0

    for _ in range(args.runs):
        t0 = time.perf_counter()
        mask, score = sam3_service._get_mask(image, args.prompt)
        _sync_cuda()
        t1 = time.perf_counter()
        _ = _postprocess(args.operation, image, mask)
        _sync_cuda()
        t2 = time.perf_counter()

        mask_times.append(t1 - t0)
        overlay_times.append(t2 - t1)
        total_times.append(t2 - t0)
        last_score = float(score)
        last_mask_pixels = int(np.count_nonzero(mask))

    result = {
        "benchmark": f"pure SAM3 mask extraction + local {args.operation}",
        "image": args.image,
        "prompt": args.prompt,
        "sam3_weight_path": args.sam3_weight_path,
        "device": args.device,
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "operation": args.operation,
        "vlm_bbox_localizer": "disabled",
        "last_score": last_score,
        "last_mask_pixels": last_mask_pixels,
        "sam3_mask": _stats(mask_times),
        f"{args.operation}_postprocess": _stats(overlay_times),
        f"sam3_mask_plus_{args.operation}": _stats(total_times),
    }

    print(json.dumps(result, indent=2))
    out_path = args.out or f"/mnt/disk1/shared_data/lzy/SOMA-local/src/outputs/latency_benchmarks/sam3_core_{args.operation}_latency.json"
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("saved_to=", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Benchmark VLM latency for SOMA planning and visual-tool localization.

Prompt-Refiner and Chaining-Step are dominated by the VLM planning call, so this
script measures `Qwen3VLAPIClient.orchestrate_perception` under prompts that
encourage each tool family. Visual Overlay and Remove Distractor have two VLM
paths: a planning/confirmation call and an optional `detect_object` bbox call
used by the SAM3 service.
"""

import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from soma_vlm import Qwen3VLAPIClient


ORCHESTRATION_SYSTEM_PROMPT = (
    "You are the visual perception cortex of a robot manipulation system (SOMA). "
    "Given a robot observation, task instruction, and historical context, decide "
    "whether perception intervention is needed. Available tools are "
    "remove_distractor, visual_overlay, instruction_refine, task_decompose, and encore. "
    "Return JSON only with keys: reasoning, refined_task, tool_chain, params."
)


SCENARIOS = {
    "generic": {
        "mode": "orchestrate",
        "task": "pick up the central black bowl and place it on the plate",
        "rag_context": "Past failures indicate visual ambiguity and distractor interference.",
    },
    "visual_overlay": {
        "mode": "orchestrate",
        "task": "pick up the central black bowl and place it on the plate",
        "rag_context": "Past failures indicate visual ambiguity among similar black bowls. Prefer visual_overlay to highlight the central target bowl.",
    },
    "remove_distractor": {
        "mode": "orchestrate",
        "task": "pick up the central black bowl and place it on the plate",
        "rag_context": "Past failures indicate distractor interference from multiple non-target black bowls. Prefer remove_distractor and list each non-target bowl individually.",
    },
    "prompt_refiner": {
        "mode": "orchestrate",
        "task": "Umm... can you grab that dark round thing in the middle, you know, the bowl, and put it on the plate?",
        "rag_context": "Past failures indicate linguistic ambiguity and noisy informal instructions. Prefer prompt refinement when visual input is clear.",
    },
    "chainstep": {
        "mode": "orchestrate",
        "task": "Pick up the cream cheese and place it in the basket, then pick up the milk and place it in the basket, then pick up the chocolate pudding and place it in the basket.",
        "rag_context": "Past failures indicate long-horizon compounding errors. Prefer task_decompose into short subtasks with control-flow reset.",
    },
    "visual_overlay_detect": {
        "mode": "detect_object",
        "prompt": "central black bowl",
    },
    "remove_distractor_detect": {
        "mode": "detect_object",
        "prompt": "non-target black bowl at the top left",
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


def _load_or_create_image(image_path: str) -> tuple[np.ndarray, str]:
    path = Path(image_path)
    if path.exists():
        return np.array(Image.open(path).convert("RGB")), str(path)

    image = Image.new("RGB", (640, 480), (225, 225, 218))
    draw = ImageDraw.Draw(image)
    draw.ellipse((55, 170, 205, 320), fill=(238, 238, 224), outline=(120, 120, 110), width=4)
    for cx, cy, r in [(420, 120, 45), (320, 170, 42), (500, 190, 44), (360, 325, 43), (500, 330, 42)]:
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(22, 22, 22), outline=(70, 70, 70), width=3)
    draw.ellipse((420 - 50, 245 - 50, 420 + 50, 245 + 50), fill=(12, 12, 12), outline=(0, 180, 80), width=5)
    return np.array(image), "synthetic_bowl_scene"


def _to_pil(image: np.ndarray) -> Image.Image:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(image).convert("RGB")


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        pass
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        text = text[start : end + 1]
    try:
        return json.loads(text.strip())
    except Exception:
        return {"raw_text": text}


class LocalQwen3VLClient:
    """Minimal local Qwen3-VL client for latency benchmarking."""

    def __init__(self, model_id: str):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.model_id = model_id
        self.base_url = "local_hf_transformers"
        model_kwargs = {
            "dtype": "auto",
            "device_map": "auto",
        }
        attn_impl = os.environ.get("SOMA_VLM_ATTN_IMPLEMENTATION", "")
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_id)

    def _generate(self, messages: list[dict], max_tokens: int = 256) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        with self.torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def orchestrate_perception(self, image: np.ndarray, task_desc: str, rag_context: str = "", rag_hints=None) -> dict:
        user_prompt = (
            f"Current Task: {task_desc}\n"
            f"Context: {rag_context}\n"
            "Analyze the image and return the SOMA intervention plan as JSON only."
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": ORCHESTRATION_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": _to_pil(image)},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        return _extract_json(self._generate(messages, max_tokens=int(os.environ.get("SOMA_VLM_MAX_NEW_TOKENS", "256"))))

    def detect_object(self, image: np.ndarray, prompt: str):
        user_prompt = (
            f"Detect the bounding box for: '{prompt}'. "
            "Return ONLY the box coordinates in JSON format: [ymin, xmin, ymax, xmax] normalized from 0 to 1000."
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a robotic vision assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": _to_pil(image)},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        text = self._generate(messages, max_tokens=64)
        numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
        if len(numbers) < 4:
            return {"raw_text": text, "bbox": None}
        h, w = image.shape[:2]
        ymin, xmin, ymax, xmax = numbers[:4]
        return {
            "raw_text": text,
            "bbox": [
                int(xmin / 1000 * w),
                int(ymin / 1000 * h),
                int(xmax / 1000 * w),
                int(ymax / 1000 * h),
            ],
        }


def _make_vlm_client():
    backend = os.environ.get("SOMA_VLM_BACKEND", "api").lower()
    model_id = os.environ.get("SOMA_VLM_MODEL_ID", "qwen3-vl-32b-instruct")
    if backend == "hf":
        return LocalQwen3VLClient(model_id), backend
    return (
        Qwen3VLAPIClient(
            api_key=os.environ.get("SOMA_VLM_API_KEY"),
            base_url=os.environ.get("SOMA_VLM_BASE_URL"),
            model_id=model_id,
        ),
        backend,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    image, image_source = _load_or_create_image(args.image)
    scenario = SCENARIOS[args.scenario]

    vlm, backend = _make_vlm_client()

    last_output = None
    for _ in range(args.warmup):
        if scenario["mode"] == "detect_object":
            last_output = vlm.detect_object(image=image, prompt=scenario["prompt"])
        else:
            last_output = vlm.orchestrate_perception(
                image=image,
                task_desc=scenario["task"],
                rag_context=scenario["rag_context"],
                rag_hints={},
            )

    times: list[float] = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        if scenario["mode"] == "detect_object":
            last_output = vlm.detect_object(image=image, prompt=scenario["prompt"])
        else:
            last_output = vlm.orchestrate_perception(
                image=image,
                task_desc=scenario["task"],
                rag_context=scenario["rag_context"],
                rag_hints={},
            )
        times.append(time.perf_counter() - t0)

    result = {
        "benchmark": f"VLM {scenario['mode']}: {args.scenario}",
        "scenario": args.scenario,
        "mode": scenario["mode"],
        "image": image_source,
        "requested_image": args.image,
        "task": scenario.get("task"),
        "prompt": scenario.get("prompt"),
        "rag_context": scenario.get("rag_context"),
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "backend": backend,
        "model_id": vlm.model_id,
        "base_url": vlm.base_url,
        "latency": _stats(times),
        "last_output": last_output,
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

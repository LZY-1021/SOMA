# SOMA Latency Benchmark Summary

Date: 2026-07-06

## Summary Table

| Component | Trigger / Scope | Runs | Mean | Median | P95 | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| RAG retrieval | Task start, 100 success + 100 failure memory bank | 300 | 4.93 ms | 4.92 ms | 5.01 ms | Returns top-1 success + top-2 failure memories. |
| VLM orchestration, generic | Task-level planning, Qwen3-VL-32B API | 40 | 7.20 s | 5.45 s | 8.65 s | Generic SOMA orchestration measurement. |
| VLM orchestration: Prompt-Refiner | Task-level planning, Qwen3-VL-32B API | 100 | 6.51 s | 4.24 s | 21.08 s | Cloud/API latency includes network and provider scheduling. |
| VLM orchestration: Chaining-Step | Task-level planning, Qwen3-VL-32B API | 100 | 13.01 s | 9.67 s | 29.13 s | Longer prompt/output; last plan also selected visual overlay. |
| SAM3-backed MCP visual tool, overlay e2e | HTTP MCP end-to-end | 180 | 1.59 s | 1.14 s | 3.74 s | Includes service overhead and auxiliary localization. |
| SAM3-backed MCP visual tool, remove e2e | HTTP MCP end-to-end | 200 | 1.36 s | 0.91 s | 2.66 s | P99 has long tail: 13.05 s. |
| Pure SAM3 mask extraction | Local, VLM localizer disabled | 200 | 159.63 ms | 159.62 ms | 159.87 ms | Overlay run; prompt produced empty final mask. |
| Local overlay postprocess | Local NumPy overlay only | 200 | 1.61 ms | 1.61 ms | 1.66 ms | Measured after SAM3 mask extraction. |
| Pure SAM3 mask + overlay | Local, VLM localizer disabled | 200 | 161.24 ms | 161.23 ms | 161.51 ms | Prompt produced empty final mask. |
| Pure SAM3 mask extraction | Local, VLM localizer disabled | 200 | 159.32 ms | 159.30 ms | 159.49 ms | Remove run; prompt produced empty final mask. |
| Local remove/inpaint postprocess | Local OpenCV inpaint only | 200 | 0.32 ms | 0.32 ms | 0.33 ms | Empty mask, so this is optimistic for real removal. |
| Pure SAM3 mask + remove | Local, VLM localizer disabled | 200 | 159.63 ms | 159.61 ms | 159.80 ms | Empty mask, so remove postprocess should be rerun with non-empty mask if emphasized. |

## Key Takeaways

1. RAG retrieval is negligible: with 200 memory entries, retrieval takes about 5 ms at both mean and P95.
2. VLM orchestration is the dominant online cost when using the remote Qwen3-VL-32B API.
3. Pure local SAM3 mask extraction is about 160 ms, confirming that the 1--2 s MCP visual-tool latency is not pure SAM3 segmentation latency.
4. MCP visual-tool e2e latency includes HTTP/base64 serialization, service-side logic, optional VLM-assisted localization, SAM3 mask extraction, and image editing.
5. The current code invokes SOMA perception sparsely at task start (`step == 0`), so these costs should be reported as task-level intervention latency and optionally amortized over episode steps, not as per-step control latency.

## Caveats

- The pure SAM3 overlay and remove runs reported `last_mask_pixels=0`; the latency is still useful for the SAM3 text-prompt path, but remove/inpaint post-processing should be rerun with a prompt that yields a non-empty mask if we want a faithful non-empty-mask remove cost.
- Prompt-Refiner and Chaining-Step are VLM-dominated tools. Their measured latency reflects the remote Qwen3-VL-32B API backend, including network and provider-side scheduling.
- Chaining-Step measurement used a longer multi-subtask instruction and generated a longer response, so its latency is expected to be higher than Prompt-Refiner.

## Paper-Ready Short Text

```text
RAG retrieval is lightweight: with a 200-entry memory bank, it takes 4.93 ms on average and 5.01 ms at P95. The dominant online cost comes from VLM orchestration, which takes 6.51 s for prompt refinement and 13.01 s for long-horizon task decomposition using the remote Qwen3-VL-32B API. For visual tools, the full SAM3-backed MCP intervention takes 1.36--1.59 s on average end-to-end, while the pure local SAM3 mask extraction plus overlay/remove path takes only about 160 ms. This shows that the MCP visual-tool latency is dominated by system-level service overhead and auxiliary localization rather than SAM3 segmentation alone.
```

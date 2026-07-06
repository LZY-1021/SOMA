# SAM3 MCP Visual Tool Latency

Date: 2026-07-06

## Setup

- Component: representative SAM3-backed MCP visual tool
- Measurement type: online blocking HTTP tool call latency
- Measured runs: 180

## Result

| Component | Trigger | Mean | Median | P95 | Blocks control loop? |
| --- | --- | ---: | ---: | ---: | --- |
| SAM3-backed MCP visual tool, end-to-end | If invoked | 1.59 s | 1.14 s | 3.74 s | Yes |
| Pure SAM3 mask extraction | If invoked | 159.63 ms | 159.62 ms | 159.87 ms | Yes |
| Local overlay postprocess | If invoked | 1.61 ms | 1.61 ms | 1.66 ms | Yes |
| Pure SAM3 mask + local overlay | If invoked | 161.24 ms | 161.23 ms | 161.51 ms | Yes |

Raw output:

```text
SAM3-backed MCP visual tool, end-to-end
n= 180
mean_s= 1.5945553668270198
median_s= 1.1415135699789971
p95_s= 3.739374295808375
```

Pure SAM3 mask + local overlay raw output:

```text
benchmark= pure SAM3 mask extraction + local overlay
warmup_runs= 25
measured_runs= 200
vlm_bbox_localizer= disabled
last_score= 0.0
last_mask_pixels= 0

sam3_mask:
mean_s= 0.15962965376093052
median_s= 0.15961738000623882
p95_s= 0.15987198986113071
p99_s= 0.15996900596655905

overlay_postprocess:
mean_s= 0.0016127282206434756
median_s= 0.0016080704517662525
p95_s= 0.0016589760780334473
p99_s= 0.0017264271154999733

sam3_mask_plus_overlay:
mean_s= 0.161242381981574
median_s= 0.16122506151441485
p95_s= 0.16150965401902795
p99_s= 0.16160397185012698
```

Note: `last_mask_pixels=0` indicates that the prompt used in this run did not produce a non-empty mask. The measured SAM3 text-prompt path is still useful for latency, but a prompt with a non-empty mask should be used when benchmarking remove/inpainting post-processing.

Paper-ready statement:

```text
The representative SAM3-backed MCP visual intervention takes 1.59 s on average and 3.74 s at P95 end-to-end, including service overhead and auxiliary localization. In contrast, pure local SAM3 mask extraction plus overlay takes 161.24 ms on average and 161.51 ms at P95. Since SOMA invokes visual interventions sparsely at the task level rather than at every control step, this cost is amortized over the full episode.
```

# VLM Orchestration Latency

Date: 2026-07-06

## Setup

- Component: SOMA VLM orchestration
- Model: `qwen3-vl-32b-instruct`
- Backend: remote OpenAI-compatible API
- Measured runs: 40

## Result

| Component | Backend | Runs | Mean | Median | P95 | Blocks control loop? |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| VLM orchestration | Qwen3-VL-32B API | 40 | 7.20 s | 5.45 s | 8.65 s | Yes |

Raw output:

```text
VLM orchestration
n= 40
mean_s= 7.195080955687445
median_s= 5.447212450555526
p95_s= 8.65061376313679
```

Paper-ready statement:

```text
Using the remote Qwen3-VL-32B API backend, SOMA's VLM orchestration takes 7.20 s on average and 8.65 s at P95. This latency is incurred only at sparse task-level intervention points in the current implementation, rather than at every low-level control step.
```

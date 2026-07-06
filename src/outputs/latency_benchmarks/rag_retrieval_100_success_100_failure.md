# RAG Retrieval Latency: 100 Success / 100 Failure Memory Bank

Date: 2026-07-06

## Setup

- Memory bank: `/mnt/disk1/shared_data/lzy/SOMA-local/src/outputs/experience_db_100bench`
- Success experiences: 100 metadata entries, vectors shape `(100, 1168)`
- Failure experiences: 100 metadata entries, vectors shape `(100, 1168)`
- Query setting: current `SOMAAgent.init_episode()` implementation
  - `success` retrieval: `top_k=1`
  - `failure` retrieval: `top_k=2`
- Returned context size from final query: `{'success': 1, 'failure': 2}`
- Timing repetitions: 100 warmup runs, followed by 300 measured runs

## Result

| Component | Memory size | Mean | Median | P95 | Blocking |
| --- | ---: | ---: | ---: | ---: | --- |
| RAG retrieval | 100 success + 100 failure | 4.93 ms | 4.92 ms | 5.01 ms | Yes |

Raw output:

```text
RAG retrieval
warmup_runs= 100
measured_runs= 300
mean_s= 0.004933029846288264
median_s= 0.00492303550709039
p95_s= 0.005010090069845319
p99_s= 0.005044851917773485
min_s= 0.004855083068832755
max_s= 0.0056080271024256945
last_ctx_sizes= {'success': 1, 'failure': 2}
```

Paper-ready statement:

```text
With a 200-entry memory bank (100 success and 100 failure experiences), RAG retrieval takes 4.93 ms on average and 5.01 ms at P95. Since the online policy uses only the top-1 success and top-2 failure memories, memory lookup is negligible compared with VLM orchestration.
```

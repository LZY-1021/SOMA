#!/usr/bin/env bash
#SBATCH -J soma_latency
#SBATCH -p gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=252G
#SBATCH -o outputs/latency_benchmarks/slurm-%j.out
#SBATCH -e outputs/latency_benchmarks/slurm-%j.err

set -euo pipefail

cd /mnt/disk1/shared_data/lzy/SOMA-local/src
mkdir -p outputs/latency_benchmarks

# Adjust these two lines to match the H200 cluster environment.
module load miniforge3/24.11 || true
source activate pytorch

export IMG="${IMG:-/mnt/disk1/shared_data/lzy/SOMA-local/src/experiment_step/remove-distractor/data/original.jpg}"
export TASK="${TASK:-pick up the central black bowl and place it on the plate}"
export MEMORY_DIR="${MEMORY_DIR:-/mnt/disk1/shared_data/lzy/SOMA-local/src/outputs/experience_db_100bench}"
export SAM3_WEIGHT="${SAM3_WEIGHT:-/mnt/disk1/shared_data/lzy/models/sam/sam3.pt}"
export SAM3_URL="${SAM3_URL:-http://127.0.0.1:5001}"
export DEVICE="${DEVICE:-cuda}"

export RAG_WARMUP="${RAG_WARMUP:-100}"
export RAG_RUNS="${RAG_RUNS:-300}"
export SAM3_WARMUP="${SAM3_WARMUP:-25}"
export SAM3_RUNS="${SAM3_RUNS:-200}"
export MCP_WARMUP="${MCP_WARMUP:-25}"
export MCP_RUNS="${MCP_RUNS:-200}"
export VLM_WARMUP="${VLM_WARMUP:-15}"
export VLM_RUNS="${VLM_RUNS:-100}"

export RUN_RAG="${RUN_RAG:-1}"
export RUN_VLM="${RUN_VLM:-1}"
export RUN_SAM3_CORE="${RUN_SAM3_CORE:-1}"
export RUN_MCP="${RUN_MCP:-1}"

SAM3_PID=""
cleanup() {
  if [[ -n "${SAM3_PID}" ]] && kill -0 "${SAM3_PID}" 2>/dev/null; then
    kill "${SAM3_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "${RUN_MCP}" == "1" ]]; then
  python sam3_service.py \
    --host 127.0.0.1 \
    --port 5001 \
    --device "${DEVICE}" \
    --sam3_weight_path "${SAM3_WEIGHT}" \
    > outputs/latency_benchmarks/sam3_service_h200.log 2>&1 &
  SAM3_PID="$!"

  for _ in $(seq 1 120); do
    if curl -fsS "${SAM3_URL}/health" >/dev/null; then
      break
    fi
    sleep 1
  done
fi

bash run_h200_latency_benchmarks.sh

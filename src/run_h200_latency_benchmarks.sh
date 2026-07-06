#!/usr/bin/env bash
set -euo pipefail

# H200 VLM latency benchmark runner for SOMA.
#
# Usage:
#   cd /mnt/disk1/shared_data/lzy/SOMA-local/src
#   sbatch -p gpu_h200 --gpus=1 ./run_h200_latency_benchmarks.sh
#
# Optional environment overrides:
#   IMG=/path/to/image.jpg
#   TASK="pick up ..."
#   MEMORY_DIR=/path/to/experience_db_100bench
#   SAM3_WEIGHT=/path/to/sam3.pt
#   SAM3_URL=http://127.0.0.1:5001
#   DEVICE=cuda
#   RUN_RAG=1 RUN_VLM=1 RUN_SAM3_CORE=1 RUN_MCP=1
#   SOMA_VLM_BASE_URL=http://127.0.0.1:8000/v1
#   SOMA_VLM_API_KEY=EMPTY
#   SOMA_VLM_MODEL_ID=qwen3-vl-32b-instruct
#
# Notes:
#   - RUN_MCP requires sam3_service.py already running at SAM3_URL.
#   - Default H200 mode only runs VLM benchmarks. SAM3/RAG/MCP are disabled.

module load miniforge3/24.11 || true
source activate "${CONDA_ENV:-qwen3vl}"

export SOMA_VLM_BASE_URL="${SOMA_VLM_BASE_URL:-http://127.0.0.1:8000/v1}"
export SOMA_VLM_API_KEY="${SOMA_VLM_API_KEY:-EMPTY}"
export SOMA_VLM_MODEL_ID="${SOMA_VLM_MODEL_ID:-qwen3-vl-32b-instruct}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

OUT_DIR="${OUT_DIR:-${ROOT_DIR}/outputs/latency_benchmarks}"
mkdir -p "${OUT_DIR}"

IMG="${IMG:-${ROOT_DIR}/experiment_step/remove-distractor/data/original.jpg}"
TASK="${TASK:-pick up the central black bowl and place it on the plate}"
MEMORY_DIR="${MEMORY_DIR:-${ROOT_DIR}/outputs/experience_db_100bench}"
SAM3_WEIGHT="${SAM3_WEIGHT:-/mnt/disk1/shared_data/lzy/models/sam/sam3.pt}"
SAM3_URL="${SAM3_URL:-http://127.0.0.1:5001}"
DEVICE="${DEVICE:-cuda}"

RAG_WARMUP="${RAG_WARMUP:-100}"
RAG_RUNS="${RAG_RUNS:-300}"
SAM3_WARMUP="${SAM3_WARMUP:-25}"
SAM3_RUNS="${SAM3_RUNS:-200}"
MCP_WARMUP="${MCP_WARMUP:-25}"
MCP_RUNS="${MCP_RUNS:-200}"
VLM_WARMUP="${VLM_WARMUP:-15}"
VLM_RUNS="${VLM_RUNS:-100}"
VLM_SCENARIOS="${VLM_SCENARIOS:-generic visual_overlay_detect remove_distractor_detect prompt_refiner chainstep}"

OVERLAY_PROMPT="${OVERLAY_PROMPT:-central black bowl}"
REMOVE_PROMPT="${REMOVE_PROMPT:-black bowl}"

RUN_RAG="${RUN_RAG:-0}"
RUN_VLM="${RUN_VLM:-1}"
RUN_SAM3_CORE="${RUN_SAM3_CORE:-0}"
RUN_MCP="${RUN_MCP:-0}"

echo "[SOMA latency] root=${ROOT_DIR}"
echo "[SOMA latency] out=${OUT_DIR}"
echo "[SOMA latency] image=${IMG}"
echo "[SOMA latency] memory=${MEMORY_DIR}"
echo "[SOMA latency] sam3_url=${SAM3_URL}"
echo "[SOMA latency] vlm_base_url=${SOMA_VLM_BASE_URL}"
echo "[SOMA latency] vlm_model=${SOMA_VLM_MODEL_ID}"
echo "[SOMA latency] run_flags=rag:${RUN_RAG} vlm:${RUN_VLM} sam3_core:${RUN_SAM3_CORE} mcp:${RUN_MCP}"

if [[ "${RUN_RAG}" == "1" ]]; then
  echo "[1/4] RAG retrieval latency"
  python benchmark_rag_retrieval_latency.py \
    --image "${IMG}" \
    --task "${TASK}" \
    --memory_dir "${MEMORY_DIR}" \
    --device "${DEVICE}" \
    --sam3_url "${SAM3_URL}" \
    --warmup "${RAG_WARMUP}" \
    --runs "${RAG_RUNS}" \
    --out "${OUT_DIR}/rag_retrieval_h200.json" \
    | tee "${OUT_DIR}/rag_retrieval_h200.log"
fi

if [[ "${RUN_VLM}" == "1" ]]; then
  echo "[2/4] VLM orchestration latency"
  for scenario in ${VLM_SCENARIOS}; do
    echo "  - scenario=${scenario}"
    python benchmark_vlm_tool_latency.py \
      --image "${IMG}" \
      --scenario "${scenario}" \
      --warmup "${VLM_WARMUP}" \
      --runs "${VLM_RUNS}" \
      --out "${OUT_DIR}/vlm_${scenario}_h200.json" \
      | tee "${OUT_DIR}/vlm_${scenario}_h200.log"
  done
fi

if [[ "${RUN_SAM3_CORE}" == "1" ]]; then
  echo "[3/4] Pure SAM3 core latency"
  python benchmark_sam3_core_latency.py \
    --image "${IMG}" \
    --prompt "${OVERLAY_PROMPT}" \
    --operation overlay \
    --sam3_weight_path "${SAM3_WEIGHT}" \
    --device "${DEVICE}" \
    --warmup "${SAM3_WARMUP}" \
    --runs "${SAM3_RUNS}" \
    --out "${OUT_DIR}/sam3_core_overlay_h200.json" \
    | tee "${OUT_DIR}/sam3_core_overlay_h200.log"

  python benchmark_sam3_core_latency.py \
    --image "${IMG}" \
    --prompt "${REMOVE_PROMPT}" \
    --operation remove \
    --sam3_weight_path "${SAM3_WEIGHT}" \
    --device "${DEVICE}" \
    --warmup "${SAM3_WARMUP}" \
    --runs "${SAM3_RUNS}" \
    --out "${OUT_DIR}/sam3_core_remove_h200.json" \
    | tee "${OUT_DIR}/sam3_core_remove_h200.log"
fi

if [[ "${RUN_MCP}" == "1" ]]; then
  echo "[4/4] MCP visual tool e2e latency"
  if ! curl -fsS "${SAM3_URL}/health" >/dev/null; then
    echo "[ERROR] SAM3 service is not reachable at ${SAM3_URL}."
    echo "Start it first, for example:"
    echo "  python sam3_service.py --host 0.0.0.0 --port 5001 --device ${DEVICE} --sam3_weight_path ${SAM3_WEIGHT}"
    exit 2
  fi

  python benchmark_mcp_visual_tool_latency.py \
    --image "${IMG}" \
    --prompt "${OVERLAY_PROMPT}" \
    --operation overlay \
    --sam3_url "${SAM3_URL}" \
    --warmup "${MCP_WARMUP}" \
    --runs "${MCP_RUNS}" \
    --out "${OUT_DIR}/mcp_overlay_e2e_h200.json" \
    | tee "${OUT_DIR}/mcp_overlay_e2e_h200.log"

  python benchmark_mcp_visual_tool_latency.py \
    --image "${IMG}" \
    --prompt "${REMOVE_PROMPT}" \
    --operation remove \
    --sam3_url "${SAM3_URL}" \
    --warmup "${MCP_WARMUP}" \
    --runs "${MCP_RUNS}" \
    --out "${OUT_DIR}/mcp_remove_e2e_h200.json" \
    | tee "${OUT_DIR}/mcp_remove_e2e_h200.log"
fi

echo "[SOMA latency] done. Results written to ${OUT_DIR}"

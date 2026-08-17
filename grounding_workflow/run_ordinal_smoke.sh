#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${ORDINAL_CONFIG:-${ROOT}/ordinal.env}"
[[ -f "${CONFIG}" ]] || {
  echo "找不到配置文件: ${CONFIG}"
  echo "先执行: cp ordinal.env.example ordinal.env"
  exit 1
}
source "${CONFIG}"

: "${QWEN_MODEL:?ordinal.env 缺少 QWEN_MODEL}"
: "${LOCATEANYTHING_MODEL:?ordinal.env 缺少 LOCATEANYTHING_MODEL}"
: "${DATA_DIR:?ordinal.env 缺少 DATA_DIR}"
: "${OUTPUT_ROOT:?ordinal.env 缺少 OUTPUT_ROOT}"

QUERY_FILE="${QUERY_FILE:-${DATA_DIR}/queries/queries.json}"
RUN_NAME="${RUN_NAME:-ordinal-all}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

ARGS=(
  --qwen_model "${QWEN_MODEL}"
  --locateanything_model "${LOCATEANYTHING_MODEL}"
  --data_dir "${DATA_DIR}"
  --query_file "${QUERY_FILE}"
  --output "${RUN_DIR}/predictions.json"
  --raw_output "${RUN_DIR}/raw.jsonl"
  --limit "${LIMIT:-0}"
  --qwen_dtype "${QWEN_DTYPE:-auto}"
  --qwen_max_new_tokens "${QWEN_MAX_NEW_TOKENS:-128}"
  --locateanything_max_new_tokens "${LOCATEANYTHING_MAX_NEW_TOKENS:-2048}"
  --attn "${LOCATEANYTHING_ATTN:-sdpa}"
  --vision_attn "${LOCATEANYTHING_VISION_ATTN:-auto}"
  --scheduler "${LOCATEANYTHING_SCHEDULER:-eager}"
  --group_size "${LOCATEANYTHING_GROUP_SIZE:-0}"
  --feature_cache_size "${LOCATEANYTHING_FEATURE_CACHE_SIZE:-1}"
)

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${GPU_IDS:-}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" python3 "${ROOT}/src/run_ordinal_workflow.py" "${ARGS[@]}"
else
  python3 "${ROOT}/src/run_ordinal_workflow.py" "${ARGS[@]}"
fi

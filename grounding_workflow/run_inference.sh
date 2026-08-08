#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:?用法: run_inference.sh MODEL DATA_DIR OUTPUT [BACKEND]}"
DATA_DIR="${2:?缺少 DATA_DIR}"
OUTPUT="${3:?缺少 OUTPUT}"
BACKEND="${4:-${MODEL_BACKEND:-qwen}}"
QUERY_FILE="${DATA_DIR}/queries/queries.json"

ARGS=(--backend "${BACKEND}" --model "${MODEL}" --data_dir "${DATA_DIR}" --query_file "${QUERY_FILE}" --output "${OUTPUT}" --batch_size "${BATCH_SIZE:-4}")
[[ -n "${LIMIT:-}" ]] && ARGS+=(--limit "${LIMIT}")
[[ "${NO_RESUME:-0}" == "1" ]] && ARGS+=(--no_resume)
[[ -n "${PROMPT_PROVIDER:-}" ]] && ARGS+=(--prompt_provider "${PROMPT_PROVIDER}")
[[ -n "${PROMPT_PREFIX:-}" ]] && ARGS+=(--prompt_prefix "${PROMPT_PREFIX}")
if [[ "${BACKEND}" == "locateanything" ]]; then
  ARGS+=(--attn "${LOCATEANYTHING_ATTN:-sdpa}" --vision_attn "${LOCATEANYTHING_VISION_ATTN:-auto}" --scheduler "${LOCATEANYTHING_SCHEDULER:-eager}" --group_size "${LOCATEANYTHING_GROUP_SIZE:-0}" --max_new_tokens "${LOCATEANYTHING_MAX_NEW_TOKENS:-2048}" --save_every "${SAVE_EVERY:-100}")
else
  ARGS+=(--max_new_tokens "${QWEN_MAX_NEW_TOKENS:-128}")
fi
PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" python3 "${ROOT}/src/inference.py" "${ARGS[@]}"
PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" python3 "${ROOT}/src/validate_submission.py" --predictions "${OUTPUT}" --queries "${QUERY_FILE}"
PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" python3 "${ROOT}/src/package_submission.py" --predictions "${OUTPUT}" --queries "${QUERY_FILE}" --output "${SUBMISSION_ZIP:-${ROOT}/submission.zip}"

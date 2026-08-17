#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:?用法: run_inference.sh MODEL DATA_DIR OUTPUT [BACKEND]}"
DATA_DIR="${2:?缺少 DATA_DIR}"
OUTPUT="${3:?缺少 OUTPUT}"
BACKEND="${4:-${MODEL_BACKEND:-qwen}}"
QUERY_FILE="${DATA_DIR}/queries/queries.json"

ARGS=(--backend "${BACKEND}" --model "${MODEL}" --data_dir "${DATA_DIR}" --query_file "${QUERY_FILE}" --output "${OUTPUT}" --batch_size "${BATCH_SIZE:-4}")
PROMPT_PROFILE_VALUE="${PROMPT_PROFILE:-}"
if [[ -z "${PROMPT_PROFILE_VALUE}" ]]; then
  case "${BACKEND}" in
    qwen) PROMPT_PROFILE_VALUE="qwen_json" ;;
    locateanything) PROMPT_PROFILE_VALUE="locateanything" ;;
    internvl) PROMPT_PROFILE_VALUE="internvl_json_union" ;;
  esac
fi
ARGS+=(--prompt_profile "${PROMPT_PROFILE_VALUE}")
[[ -n "${LIMIT:-}" ]] && ARGS+=(--limit "${LIMIT}")
[[ "${NO_RESUME:-0}" == "1" ]] && ARGS+=(--no_resume)
[[ -n "${PROMPT_PROVIDER:-}" ]] && ARGS+=(--prompt_provider "${PROMPT_PROVIDER}")
if [[ "${BACKEND}" == "locateanything" ]]; then
  ARGS+=(--attn "${LOCATEANYTHING_ATTN:-sdpa}" --vision_attn "${LOCATEANYTHING_VISION_ATTN:-auto}" --scheduler "${LOCATEANYTHING_SCHEDULER:-eager}" --group_size "${LOCATEANYTHING_GROUP_SIZE:-0}" --feature_cache_size "${LOCATEANYTHING_FEATURE_CACHE_SIZE:-1}" --max_new_tokens "${LOCATEANYTHING_MAX_NEW_TOKENS:-2048}" --save_every "${SAVE_EVERY:-100}")
elif [[ "${BACKEND}" == "internvl" ]]; then
  ARGS+=(--internvl_dtype "${INTERNVL_DTYPE:-bfloat16}" --internvl_image_cache_size "${INTERNVL_IMAGE_CACHE_SIZE:-1}" --max_new_tokens "${INTERNVL_MAX_NEW_TOKENS:-128}")
else
  ARGS+=(--max_new_tokens "${QWEN_MAX_NEW_TOKENS:-128}")
fi

run_one_worker() {
  local gpu="$1"
  local index="$2"
  local count="$3"
  local worker_output="${OUTPUT}.gpu${index}.json"
  local worker_args=("${ARGS[@]}" --output "${worker_output}" --raw_output "${OUTPUT}.gpu${index}.raw.jsonl" --partial "${OUTPUT}.gpu${index}.partial.json" --prompt_cache "${OUTPUT}.gpu${index}.prompts.jsonl" --log_file "${worker_output}.log" --shard_index "${index}" --shard_count "${count}")
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" python3 "${ROOT}/src/inference.py" "${worker_args[@]}"
}

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS:-}"
if [[ "${BACKEND}" == "locateanything" && ${#GPU_LIST[@]} -gt 1 ]]; then
  WORKER_OUTPUTS=()
  WORKER_PIDS=()
  for index in "${!GPU_LIST[@]}"; do
    worker_output="${OUTPUT}.gpu${index}.json"
    WORKER_OUTPUTS+=("${worker_output}")
    run_one_worker "${GPU_LIST[$index]}" "${index}" "${#GPU_LIST[@]}" &
    WORKER_PIDS+=("$!")
  done
  for pid in "${WORKER_PIDS[@]}"; do
    wait "${pid}"
  done
  PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" python3 "${ROOT}/src/merge_predictions.py" --queries "${QUERY_FILE}" --output "${OUTPUT}" "${WORKER_OUTPUTS[@]}"
else
  PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" python3 "${ROOT}/src/inference.py" "${ARGS[@]}" --output "${OUTPUT}"
fi
PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" python3 "${ROOT}/src/validate_submission.py" --predictions "${OUTPUT}" --queries "${QUERY_FILE}"
PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" python3 "${ROOT}/src/package_submission.py" --predictions "${OUTPUT}" --queries "${QUERY_FILE}" --output "${SUBMISSION_ZIP:-${ROOT}/submission.zip}"

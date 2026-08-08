#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${BASELINE_CONFIG:-${ROOT}/server.env}"
[[ -f "${CONFIG}" ]] || { echo "找不到配置文件: ${CONFIG}"; exit 1; }
source "${CONFIG}"
cd "${ROOT}"
: "${MODEL_BACKEND:?server.env 缺少 MODEL_BACKEND}"
: "${MODEL:?server.env 缺少 MODEL}"
: "${DATA_DIR:?server.env 缺少 DATA_DIR}"
: "${OUTPUT_ROOT:?server.env 缺少 OUTPUT_ROOT}"
: "${RUN_NAME:?server.env 缺少 RUN_NAME}"
: "${BATCH_SIZE:?server.env 缺少 BATCH_SIZE}"
export MODEL_BACKEND PROMPT_PROVIDER PROMPT_PREFIX
export LOCATEANYTHING_ATTN LOCATEANYTHING_VISION_ATTN LOCATEANYTHING_SCHEDULER LOCATEANYTHING_GROUP_SIZE LOCATEANYTHING_MAX_NEW_TOKENS
export GPU_IDS
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
QUERY_FILE="${DATA_DIR}/queries/queries.json"

case "${1:-help}" in
  check)
    [[ -f "${QUERY_FILE}" ]] || { echo "找不到查询文件: ${QUERY_FILE}"; exit 1; }
    PYTHONPATH="${ROOT}/src" python3 -m compileall -q "${ROOT}/src"
    if [[ "${MODEL_BACKEND}" == "locateanything" ]]; then
      python3 -c "import cv2, decord, lmdb"
      python3 -c "import transformers, sys; major=int(transformers.__version__.split('.')[0]); assert major < 5, f'LocateAnything 需要 Transformers 4.x，当前为 {transformers.__version__}'"
    fi
    echo "配置、查询文件和 Python 模块检查通过" ;;
  smoke)
    mkdir -p "${OUTPUT_ROOT}/smoke/${MODEL_BACKEND}"
    BATCH_SIZE="${BATCH_SIZE}" LIMIT="${SMOKE_LIMIT}" NO_RESUME=1 bash "${ROOT}/run_inference.sh" "${MODEL}" "${DATA_DIR}" "${OUTPUT_ROOT}/smoke/${MODEL_BACKEND}/predictions.json" "${MODEL_BACKEND}" ;;
  full|resume)
    mkdir -p "${RUN_DIR}"
    BATCH_SIZE="${BATCH_SIZE}" SUBMISSION_ZIP="${RUN_DIR}/submission.zip" bash "${ROOT}/run_inference.sh" "${MODEL}" "${DATA_DIR}" "${RUN_DIR}/predictions.json" "${MODEL_BACKEND}" ;;
  validate)
    PYTHONPATH="${ROOT}/src" python3 "${ROOT}/src/validate_submission.py" --predictions "${RUN_DIR}/predictions.json" --queries "${QUERY_FILE}" ;;
  package)
    PYTHONPATH="${ROOT}/src" python3 "${ROOT}/src/package_submission.py" --predictions "${RUN_DIR}/predictions.json" --queries "${QUERY_FILE}" --output "${RUN_DIR}/submission.zip" ;;
  *) echo "用法: bash run_server.sh {check|smoke|full|resume|validate|package}"; exit 1 ;;
esac

#!/bin/sh
# entrypoint for running the UI dashboard inside docker-compose
set -eu

API_URL="${API_URL:-http://api:5000}"
HOST="${UI_HOST:-0.0.0.0}"
PORT="${UI_PORT:-7860}"
SHARE="${UI_SHARE:-false}"

if [ -n "${CLASSIFICATION_MODEL_PATH:-}" ]; then
  CLASS_MODEL="${CLASSIFICATION_MODEL_PATH}"
elif CLASS_MODEL=$(find /app/models -maxdepth 1 -type f -name '*classification*.joblib' ! -name '*survival*' 2>/dev/null | head -n 1); then
  echo "auto-detected classification model: ${CLASS_MODEL}"
else
  CLASS_MODEL=""
fi

if [ -n "${SURVIVAL_MODEL_PATH:-}" ]; then
  SURV_MODEL="${SURVIVAL_MODEL_PATH}"
elif SURV_MODEL=$(find /app/models -maxdepth 1 -type f -name '*survival*.joblib' 2>/dev/null | head -n 1); then
  echo "auto-detected survival model: ${SURV_MODEL}"
else
  SURV_MODEL=""
fi

CLASS_ARG=""
if [ -n "${CLASS_MODEL}" ]; then
  CLASS_ARG="--classification-model ${CLASS_MODEL}"
fi

SURV_ARG=""
if [ -n "${SURV_MODEL}" ]; then
  SURV_ARG="--survival-model ${SURV_MODEL}"
fi

SHARE_FLAG=""
if [ "${SHARE}" = "true" ] || [ "${SHARE}" = "1" ]; then
  SHARE_FLAG="--share"
fi

echo "starting ui dashboard on ${HOST}:${PORT} (api=${API_URL})"
exec python scripts/run_ui.py \
  --api-url "${API_URL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  ${SHARE_FLAG} \
  ${CLASS_ARG} \
  ${SURV_ARG}

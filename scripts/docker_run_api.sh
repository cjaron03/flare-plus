#!/bin/sh
# entrypoint for running the API server inside docker-compose
set -euo pipefail

HOST="${API_HOST:-0.0.0.0}"
PORT="${API_PORT:-5000}"
WORKERS="${API_WORKERS:-2}"

# prefer explicit model paths, otherwise auto-discover from mounted volume
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

echo "starting api server on ${HOST}:${PORT} (workers=${WORKERS})"
exec python scripts/run_api_server.py \
  --host "${HOST}" \
  --port "${PORT}" \
  --workers "${WORKERS}" \
  ${CLASS_ARG} \
  ${SURV_ARG}

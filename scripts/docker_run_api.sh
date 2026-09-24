#!/bin/sh
# entrypoint for running the API server inside docker-compose
set -eu

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

ERROR_LOG="/tmp/flare_api_start.err"
CHILD_PID=""

forward_signal() {
  if [ -n "${CHILD_PID}" ]; then
    kill "-$1" "${CHILD_PID}" 2>/dev/null || true
  fi
}

run_api_server() {
  python scripts/run_api_server.py \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers "${WORKERS}" \
    ${CLASS_ARG} \
    ${SURV_ARG} \
    2>"${ERROR_LOG}" &
  CHILD_PID=$!

  trap 'forward_signal TERM' TERM
  trap 'forward_signal INT' INT

  if wait "${CHILD_PID}"; then
    trap - TERM INT
    CHILD_PID=""
    return 0
  fi

  status=$?
  trap - TERM INT
  CHILD_PID=""
  return "${status}"
}

if run_api_server; then
  exit 0
fi
status=$?

if grep -q "Resource deadlock avoided" "${ERROR_LOG}" 2>/dev/null; then
  echo "detected bind-mount file lock, retrying api from runtime copy"
  RUNTIME_ROOT="/tmp/flare-runtime-api"
  rm -rf "${RUNTIME_ROOT}"
  mkdir -p "${RUNTIME_ROOT}"
  cp -R /app/src "${RUNTIME_ROOT}/src"
  cp -R /app/scripts "${RUNTIME_ROOT}/scripts"
  if [ -f /app/config.yaml ]; then
    cp /app/config.yaml "${RUNTIME_ROOT}/config.yaml"
  fi
  export PYTHONPATH="${RUNTIME_ROOT}"
  cd "${RUNTIME_ROOT}"
  exec python scripts/run_api_server.py \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers "${WORKERS}" \
    ${CLASS_ARG} \
    ${SURV_ARG}
fi

cat "${ERROR_LOG}" >&2
exit "${status}"

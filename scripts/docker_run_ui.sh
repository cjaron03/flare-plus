#!/bin/sh
# entrypoint for running the UI dashboard inside docker-compose
set -eu

API_URL="${API_URL:-http://api:5000}"
HOST="${UI_HOST:-0.0.0.0}"
PORT="${UI_PORT:-7860}"
STATIC_DIR="${UI_STATIC_DIR:-/app/ui-frontend/dist}"

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

if [ ! -d "${STATIC_DIR}" ]; then
  echo "warning: svelte build not found at ${STATIC_DIR}"
  echo "run 'npm install && npm run build' inside ui-frontend/ on the host and re-run ./flare ui"
fi

echo "starting ui dashboard on ${HOST}:${PORT} (api=${API_URL}, static=${STATIC_DIR})"

ERROR_LOG="/tmp/flare_ui_start.err"
if python scripts/run_ui.py \
  --api-url "${API_URL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --static-dir "${STATIC_DIR}" \
  ${CLASS_ARG} \
  ${SURV_ARG} \
  2>"${ERROR_LOG}"; then
  exit 0
fi

if grep -q "Resource deadlock avoided" "${ERROR_LOG}" 2>/dev/null; then
  echo "detected bind-mount file lock, retrying ui from runtime copy"
  RUNTIME_ROOT="/tmp/flare-runtime-ui"
  rm -rf "${RUNTIME_ROOT}"
  mkdir -p "${RUNTIME_ROOT}"
  cp -R /app/src "${RUNTIME_ROOT}/src"
  cp -R /app/scripts "${RUNTIME_ROOT}/scripts"
  if [ -f /app/config.yaml ]; then
    cp /app/config.yaml "${RUNTIME_ROOT}/config.yaml"
  fi
  export PYTHONPATH="${RUNTIME_ROOT}"
  cd "${RUNTIME_ROOT}"
  exec python scripts/run_ui.py \
    --api-url "${API_URL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --static-dir "${STATIC_DIR}" \
    ${CLASS_ARG} \
    ${SURV_ARG}
fi

cat "${ERROR_LOG}" >&2
exit 1

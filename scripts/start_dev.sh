#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/AI_POC/venvs/debalekha/bin/python}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 1
fi

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "${BACKEND_PID}" 2>/dev/null || true
    wait "${BACKEND_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "${ROOT_DIR}"
HOST=0.0.0.0 \
PORT="${BACKEND_PORT}" \
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/workloadiq-matplotlib}" \
"${PYTHON_BIN}" -m workload_analysis.server_with_upload &
BACKEND_PID=$!

cd "${ROOT_DIR}/frontend"
VITE_BACKEND_PROXY="http://127.0.0.1:${BACKEND_PORT}" \
VITE_DEV_PORT="${FRONTEND_PORT}" \
npm run dev -- --host 0.0.0.0 --port "${FRONTEND_PORT}"

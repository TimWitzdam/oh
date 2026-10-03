#!/bin/sh
# One container, two runtimes: Next.js serves the UI and scores with onnxruntime,
# while a loopback-only Python service hosts the torch detector. If the Python
# side is missing or slow to start, the encoder tiers keep working.
set -eu

PYTHON_BIN="${OH_PYTHON:-/srv/venv/bin/python}"
PYTHON_SERVICE=0

log() {
  printf '%s oh %s\n' "$(date -u '+%Y/%m/%d %H:%M:%S')" "$*"
}

term() {
  log 'shutting down'
  [ -n "${node_pid:-}" ] && kill -TERM "$node_pid" 2>/dev/null || true
  [ -n "${python_pid:-}" ] && kill -TERM "$python_pid" 2>/dev/null || true
  wait 2>/dev/null || true
  exit 0
}
trap term INT TERM

if [ -x "$PYTHON_BIN" ]; then
  PYTHONPATH=/srv OH_DATA_DIR="${OH_DATA_DIR:-/data}" \
    "$PYTHON_BIN" -m uvicorn oh_service.main:app \
      --host 127.0.0.1 --port 8001 --log-level warning --no-access-log &
  python_pid=$!
  PYTHON_SERVICE=1
  log "inference service started on 127.0.0.1:8001 (pid $python_pid)"
else
  log 'python inference service not available; the Deep detector tier is disabled'
fi

cd /app
PORT="${PORT:-3000}" node server.js &
node_pid=$!
log "web app started on :${PORT} (pid $node_pid)"

wait "$node_pid"
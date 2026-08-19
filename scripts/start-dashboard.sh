#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${MEMORY_QUERY_PORT:-7748}"
PID_FILE="${HOME}/.memory/dashboard_server.pid"
LOG_FILE="${HOME}/.memory/query.log"

mkdir -p "${HOME}/.memory"

if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$EXISTING_PID" ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "Dashboard already running (pid $EXISTING_PID)"
    echo "URL: http://localhost:${PORT}"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

if command -v lsof >/dev/null 2>&1; then
  PORT_PID="$(lsof -ti tcp:"$PORT" -sTCP:LISTEN 2>/dev/null | head -n 1 || true)"
  if [[ -n "$PORT_PID" ]]; then
    echo "Port ${PORT} is already in use by pid ${PORT_PID}."
    echo "If this is the dashboard, open: http://localhost:${PORT}"
    echo "Otherwise free the port or set MEMORY_QUERY_PORT to another value."
    exit 1
  fi
fi

cd "$ROOT_DIR"
nohup python3 memory/dashboard_server.py >> "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

sleep 1
if kill -0 "$PID" 2>/dev/null; then
  echo "Dashboard started (pid $PID)"
  echo "Log: $LOG_FILE"
  echo "URL: http://localhost:${PORT}"
else
  echo "Failed to start dashboard. Check log: $LOG_FILE" >&2
  rm -f "$PID_FILE"
  exit 1
fi

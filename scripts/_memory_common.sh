#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEMORY_HOME="${HOME}/.memory"
UID_NUM="$(id -u)"

mkdir -p "$MEMORY_HOME"

all_services() {
  echo daemon ingest query
}

validate_service() {
  case "$1" in
    daemon|ingest|query) ;;
    *)
      echo "Unknown service: $1" >&2
      echo "Valid services: daemon ingest query" >&2
      return 1
      ;;
  esac
}

service_label() {
  case "$1" in
    daemon) echo "com.memory.daemon" ;;
    ingest) echo "com.memory.ingest" ;;
    query)  echo "com.memory.query" ;;
  esac
}

service_script() {
  case "$1" in
    daemon) echo "$ROOT_DIR/memory/daemon.py" ;;
    ingest) echo "$ROOT_DIR/memory/ingest_server.py" ;;
    query)  echo "$ROOT_DIR/memory/dashboard_server.py" ;;
  esac
}

service_log() {
  case "$1" in
    daemon) echo "$MEMORY_HOME/daemon.log" ;;
    ingest) echo "$MEMORY_HOME/ingest.log" ;;
    query)  echo "$MEMORY_HOME/query.log" ;;
  esac
}

service_pidfile() {
  echo "$MEMORY_HOME/$1.pid"
}

service_port() {
  case "$1" in
    ingest) echo "7747" ;;
    query)  echo "7748" ;;
    *)      echo "" ;;
  esac
}

service_plist() {
  echo "$HOME/Library/LaunchAgents/$(service_label "$1").plist"
}

service_name() {
  case "$1" in
    daemon) echo "daemon" ;;
    ingest) echo "ingest server" ;;
    query)  echo "query server" ;;
  esac
}

launchd_available() {
  command -v launchctl >/dev/null 2>&1
}

has_launchd_plist() {
  [[ -f "$(service_plist "$1")" ]]
}

launchd_loaded() {
  local label
  label="$(service_label "$1")"
  launchctl print "gui/$UID_NUM/$label" >/dev/null 2>&1
}

manual_pid() {
  local pidfile
  pidfile="$(service_pidfile "$1")"
  if [[ -f "$pidfile" ]]; then
    cat "$pidfile"
  fi
}

pid_running() {
  local pid="${1:-}"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

listening_pid() {
  local port
  port="$(service_port "$1")"
  if [[ -z "$port" ]]; then
    return 0
  fi
  if command -v lsof >/dev/null 2>&1; then
    lsof -ti "tcp:$port" -sTCP:LISTEN 2>/dev/null | head -n 1 || true
  fi
}

start_service() {
  local svc="$1"
  validate_service "$svc"

  if launchd_available && has_launchd_plist "$svc"; then
    local plist label
    plist="$(service_plist "$svc")"
    label="$(service_label "$svc")"
    if launchd_loaded "$svc"; then
      launchctl kickstart -k "gui/$UID_NUM/$label"
      echo "Started $(service_name "$svc") via launchctl (kickstart)"
    else
      launchctl bootstrap "gui/$UID_NUM" "$plist"
      echo "Started $(service_name "$svc") via launchctl (bootstrap)"
    fi
    return 0
  fi

  local pidfile pid port port_pid log script
  pidfile="$(service_pidfile "$svc")"
  pid="$(manual_pid "$svc" || true)"
  if pid_running "$pid"; then
    echo "$(service_name "$svc") already running (pid $pid)"
    return 0
  fi

  port="$(service_port "$svc")"
  port_pid="$(listening_pid "$svc")"
  if [[ -n "$port_pid" ]]; then
    echo "Cannot start $(service_name "$svc"): port $port already in use by pid $port_pid" >&2
    return 1
  fi

  log="$(service_log "$svc")"
  script="$(service_script "$svc")"

  cd "$ROOT_DIR"
  nohup python3 "$script" >> "$log" 2>&1 &
  pid=$!
  echo "$pid" > "$pidfile"
  sleep 1

  if pid_running "$pid"; then
    echo "Started $(service_name "$svc") (pid $pid)"
  else
    rm -f "$pidfile"
    echo "Failed to start $(service_name "$svc"). Check log: $log" >&2
    return 1
  fi
}

stop_service() {
  local svc="$1"
  validate_service "$svc"

  if launchd_available && has_launchd_plist "$svc"; then
    local plist
    plist="$(service_plist "$svc")"
    if launchd_loaded "$svc"; then
      launchctl bootout "gui/$UID_NUM" "$plist"
      echo "Stopped $(service_name "$svc") via launchctl"
    else
      echo "$(service_name "$svc") not loaded in launchctl"
    fi
    rm -f "$(service_pidfile "$svc")"
    return 0
  fi

  local pid pidfile port_pid
  pidfile="$(service_pidfile "$svc")"
  pid="$(manual_pid "$svc" || true)"

  if ! pid_running "$pid"; then
    port_pid="$(listening_pid "$svc")"
    if [[ -n "$port_pid" ]]; then
      pid="$port_pid"
    else
      rm -f "$pidfile"
      echo "$(service_name "$svc") not running"
      return 0
    fi
  fi

  kill "$pid" 2>/dev/null || true
  for _ in 1 2 3 4 5; do
    if ! pid_running "$pid"; then
      break
    fi
    sleep 1
  done
  if pid_running "$pid"; then
    kill -9 "$pid" 2>/dev/null || true
  fi

  rm -f "$pidfile"
  echo "Stopped $(service_name "$svc")"
}

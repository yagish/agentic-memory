#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEMORY_HOME="${HOME}/.memory"
UID_NUM="$(id -u)"

mkdir -p "$MEMORY_HOME"

all_services() {
  echo daemon recall ingest query
}

validate_service() {
  case "$1" in
    daemon|recall|ingest|query) ;;
    *)
      echo "Unknown service: $1" >&2
      echo "Valid services: daemon recall ingest query" >&2
      return 1
      ;;
  esac
}

service_label() {
  case "$1" in
    daemon) echo "com.memory.daemon" ;;
    recall|ingest) echo "com.memory.ingest" ;;
    query)  echo "com.memory.query" ;;
  esac
}

service_module() {
  case "$1" in
    daemon) echo "memory.daemon" ;;
    recall|ingest) echo "memory.servers.ingest_server" ;;
    query)  echo "memory.servers.dashboard_server" ;;
  esac
}

service_log() {
  case "$1" in
    daemon) echo "$MEMORY_HOME/daemon.log" ;;
    recall|ingest) echo "$MEMORY_HOME/ingest.log" ;;
    query)  echo "$MEMORY_HOME/query.log" ;;
  esac
}

service_pidfile() {
  echo "$MEMORY_HOME/$1.pid"
}

service_port() {
  case "$1" in
    recall|ingest) echo "${MEMORY_INGEST_PORT:-7747}" ;;
    query)  echo "${MEMORY_QUERY_PORT:-7748}" ;;
    *)      echo "" ;;
  esac
}

service_plist() {
  echo "$HOME/Library/LaunchAgents/$(service_label "$1").plist"
}

service_name() {
  case "$1" in
    daemon) echo "daemon" ;;
    recall|ingest) echo "recall server" ;;
    query)  echo "query server" ;;
  esac
}

service_debug_prefix() {
  case "$1" in
    daemon) echo "MEMORY_DEBUG_DAEMON" ;;
    recall|ingest) echo "MEMORY_DEBUG_INGEST" ;;
    query)  echo "MEMORY_DEBUG_QUERY" ;;
  esac
}
service_debug_port() {
  local prefix var_name
  prefix="$(service_debug_prefix "$1")"
  var_name="${prefix}_PORT"
  echo "${!var_name:-}"
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

  local debug_port
  debug_port="$(service_debug_port "$svc")"

  if [[ -n "$debug_port" ]] && launchd_available && has_launchd_plist "$svc" && launchd_loaded "$svc"; then
    echo "Cannot start $(service_name "$svc") in debug mode: launchctl job is already loaded. Stop it first: ./scripts/stop-memory.sh $svc" >&2
    return 1
  fi

  if [[ -z "$debug_port" ]] && launchd_available && has_launchd_plist "$svc"; then
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

  local pidfile pid port port_pid log module
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
  module="$(service_module "$svc")"

  if [[ -n "$debug_port" ]]; then
    echo "Starting $(service_name "$svc") in debug mode on ${debug_port} (direct python launch)"
  fi

  cd "$ROOT_DIR"
  nohup python3 -m "$module" >> "$log" 2>&1 &
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

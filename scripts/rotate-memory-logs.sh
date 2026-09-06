#!/usr/bin/env bash
set -euo pipefail

MEMORY_HOME="${MEMORY_HOME:-$HOME/.memory}"
ARCHIVE_DIR="${MEMORY_LOG_ARCHIVE_DIR:-$MEMORY_HOME/log-archive}"
RETENTION_DAYS="${MEMORY_LOG_RETENTION_DAYS:-7}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

case "$RETENTION_DAYS" in
  ''|*[!0-9]*)
    echo "MEMORY_LOG_RETENTION_DAYS must be a non-negative integer, got: $RETENTION_DAYS" >&2
    exit 1
    ;;
esac

mkdir -p "$MEMORY_HOME" "$ARCHIVE_DIR"

shopt -s nullglob
for log_file in "$MEMORY_HOME"/*.log; do
  [[ -f "$log_file" ]] || continue
  [[ -s "$log_file" ]] || continue

  base_name="$(basename "$log_file")"
  archive_file="$ARCHIVE_DIR/$base_name.$STAMP"

  cp -p "$log_file" "$archive_file"
  : > "$log_file"
done

find "$ARCHIVE_DIR" -type f -name '*.log.*' -mtime +"$RETENTION_DAYS" -delete

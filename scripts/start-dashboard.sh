#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_memory_common.sh"

start_service query

echo "URL: http://localhost:${MEMORY_QUERY_PORT:-7748}"

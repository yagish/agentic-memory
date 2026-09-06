#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_memory_common.sh"

services=("$@")
if [[ ${#services[@]} -eq 0 ]]; then
  services=(daemon recall query)
fi

for svc in "${services[@]}"; do
  start_service "$svc"
done

echo ""
echo "Memory system startup complete."
echo "Dashboard: http://localhost:${MEMORY_QUERY_PORT:-7748}"

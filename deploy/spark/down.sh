#!/usr/bin/env bash
# Stop the tunnel and the LiveTalking container.
set -uo pipefail
cd "$(dirname "$0")"
home="$(cd ../../.. && pwd)"
if [ -f "$home/tunnel.pid" ]; then
  kill "$(cat "$home/tunnel.pid")" 2>/dev/null && echo "tunnel stopped"
  rm -f "$home/tunnel.pid"
fi
docker compose down

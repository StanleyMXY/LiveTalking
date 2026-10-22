#!/usr/bin/env bash
# Start LiveTalking and a Cloudflare quick tunnel on the Spark and print the test URL.
set -euo pipefail
cd "$(dirname "$0")"
home="$(cd ../../.. && pwd)"          # /home/admin/avatar-demo: .env, models/, data/, hf-cache/, bin/
cf="$home/bin/cloudflared"
[ -x "$cf" ] || { echo "missing $cf (linux-arm64 build of cloudflared)"; exit 1; }

docker compose up -d

echo "waiting for :8010 (model load takes about a minute)..."
for _ in $(seq 1 150); do
  ss -ltn | grep -q ':8010 ' && break
  sleep 2
done
ss -ltn | grep -q ':8010 ' || { echo "livetalking did not open :8010"; docker logs --tail 40 livetalking; exit 1; }

if [ -f "$home/tunnel.pid" ] && kill -0 "$(cat "$home/tunnel.pid")" 2>/dev/null; then
  kill "$(cat "$home/tunnel.pid")" || true
  sleep 1
fi
# http2 (TCP): the default QUIC transport is UDP, which some networks drop
nohup "$cf" tunnel --no-autoupdate --protocol http2 --url http://localhost:8010 > "$home/tunnel.log" 2>&1 &
echo $! > "$home/tunnel.pid"

url=""
for _ in $(seq 1 40); do
  url="$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$home/tunnel.log" | head -1 || true)"
  [ -n "$url" ] && break
  sleep 2
done
[ -n "$url" ] || { echo "no tunnel URL yet; see $home/tunnel.log"; exit 1; }
echo
echo "Test URL: $url/dashboard.html"
echo "The address changes every time the tunnel restarts."

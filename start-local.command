#!/bin/zsh
set -u
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

ROOT=${0:A:h}
cd "$ROOT" || exit 1

healthy() {
  curl -fsS --max-time 2 "http://127.0.0.1:$1/api/health" 2>/dev/null | \
    python3 -c 'import json,sys; d=json.load(sys.stdin); assert isinstance(d,dict) and d.get("app") == "clipcut"' \
    >/dev/null 2>&1
}

listener_pid() {
  lsof -tiTCP:"$1" -sTCP:LISTEN 2>/dev/null | head -n 1
}

PORT=8790
for candidate in {8790..8799}; do
  if healthy "$candidate"; then
    URL="http://127.0.0.1:$candidate/"
    open "$URL"
    exit 0
  fi
done

if [[ -n $(listener_pid "$PORT") ]]; then
  PORT=0
  for candidate in {8791..8799}; do
    if [[ -z $(listener_pid "$candidate") ]]; then
      PORT=$candidate
      break
    fi
  done
  if [[ "$PORT" == 0 ]]; then
    exit 1
  fi
fi

URL="http://127.0.0.1:$PORT/"
nohup "$ROOT/.venv/bin/clipcut" serve --host 127.0.0.1 --port "$PORT" >/tmp/clipcut-workbench.log 2>&1 </dev/null &
SERVER_PID=$!

for _ in {1..50}; do
  if healthy "$PORT"; then
    open "$URL" 2>/dev/null || true
    exit 0
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    exit 1
  fi
  sleep 0.1
done

exit 0

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PORT="8080"
HOST="0.0.0.0"
WEB_DIR="web"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --web-dir)
      WEB_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

echo "Serving $WEB_DIR at http://${HOST}:${PORT} with websocket /ws (1s push)"
exec "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/run_web_server.py" --port "$PORT" --host "$HOST" --web-dir "$WEB_DIR"

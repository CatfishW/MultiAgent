#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="$ROOT_DIR/web/"
REMOTE="public-server:/www/wwwroot/ai.agaii.org/multi-agent/"
INTERVAL_S="15"
WATCH="0"
LOG_FILE="$ROOT_DIR/logs/web_sync_multi_agent.log"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-dir)
      SOURCE_DIR="$2"
      shift 2
      ;;
    --remote)
      REMOTE="$2"
      shift 2
      ;;
    --interval)
      INTERVAL_S="$2"
      shift 2
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    --watch)
      WATCH="1"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "${SOURCE_DIR: -1}" != "/" ]]; then
  SOURCE_DIR="${SOURCE_DIR}/"
fi
if [[ "${REMOTE: -1}" != "/" ]]; then
  REMOTE="${REMOTE}/"
fi

mkdir -p "$(dirname "$LOG_FILE")"

sync_once() {
  rsync -az --delete "$SOURCE_DIR" "$REMOTE"
}

if [[ "$WATCH" == "1" ]]; then
  while true; do
    now="$(date '+%Y-%m-%d %H:%M:%S')"
    if sync_once; then
      echo "[$now] sync ok: $SOURCE_DIR -> $REMOTE" | tee -a "$LOG_FILE"
    else
      echo "[$now] sync failed: $SOURCE_DIR -> $REMOTE" | tee -a "$LOG_FILE" >&2
    fi
    sleep "$INTERVAL_S"
  done
else
  sync_once
  echo "sync completed: $SOURCE_DIR -> $REMOTE"
fi

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RESULTS_DIR="artifacts/experiments/results"
LOGS_DIR="logs/experiments/sessions"
OUT_PATH="web/data/session_summary.json"
INTERVAL_S="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --logs-dir)
      LOGS_DIR="$2"
      shift 2
      ;;
    --out)
      OUT_PATH="$2"
      shift 2
      ;;
    --interval)
      INTERVAL_S="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$RESULTS_DIR" && -d "artifacts/experiments" ]]; then
  RESULTS_DIR="artifacts/experiments"
fi
if [[ ! -d "$LOGS_DIR" && -d "logs/experiments" ]]; then
  LOGS_DIR="logs/experiments"
fi

mkdir -p "$(dirname "$OUT_PATH")"

echo "Starting dashboard refresh loop"
echo "  results: $RESULTS_DIR"
echo "  logs:    $LOGS_DIR"
echo "  out:     $OUT_PATH"
echo "  every:   ${INTERVAL_S}s"

while true; do
  now="$(date '+%Y-%m-%d %H:%M:%S')"
  if .venv/bin/python scripts/build_dashboard_data.py --results-dir "$RESULTS_DIR" --logs-dir "$LOGS_DIR" --out "$OUT_PATH"; then
    echo "[$now] dashboard refreshed"
  else
    echo "[$now] dashboard refresh failed" >&2
  fi
  sleep "$INTERVAL_S"
done

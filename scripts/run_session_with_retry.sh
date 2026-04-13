#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="session"
LOG_FILE=""
MAX_ATTEMPTS="6"
SLEEP_SECONDS="30"
RESUME_ON_RETRY="0"
RESUME_ARG="--resume"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-name)
      SESSION_NAME="$2"
      shift 2
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    --max-attempts)
      MAX_ATTEMPTS="$2"
      shift 2
      ;;
    --sleep-seconds)
      SLEEP_SECONDS="$2"
      shift 2
      ;;
    --resume-on-retry)
      RESUME_ON_RETRY="1"
      shift
      ;;
    --no-resume-on-retry)
      RESUME_ON_RETRY="0"
      shift
      ;;
    --resume-arg)
      RESUME_ARG="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$LOG_FILE" ]]; then
  echo "--log-file is required" >&2
  exit 2
fi

if [[ $# -eq 0 ]]; then
  echo "Missing command after --" >&2
  exit 2
fi

if ! [[ "$MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-attempts must be a positive integer" >&2
  exit 2
fi

if ! [[ "$SLEEP_SECONDS" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "--sleep-seconds must be a non-negative number" >&2
  exit 2
fi

mkdir -p "$(dirname "$LOG_FILE")"
: >"$LOG_FILE"

base_cmd=("$@")

attempt=1
while (( attempt <= MAX_ATTEMPTS )); do
  now="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$now] session=${SESSION_NAME} attempt=${attempt}/${MAX_ATTEMPTS}" | tee -a "$LOG_FILE"

  cmd=("${base_cmd[@]}")
  if (( attempt > 1 )) && [[ "$RESUME_ON_RETRY" == "1" ]]; then
    cmd+=("$RESUME_ARG")
    echo "[$now] session=${SESSION_NAME} retry-mode=resume arg=${RESUME_ARG}" | tee -a "$LOG_FILE"
  fi

  set +e
  "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
  cmd_rc=${PIPESTATUS[0]}
  set -e

  now="$(date '+%Y-%m-%d %H:%M:%S')"
  if [[ $cmd_rc -eq 0 ]]; then
    echo "[$now] session=${SESSION_NAME} succeeded on attempt ${attempt}/${MAX_ATTEMPTS}" | tee -a "$LOG_FILE"
    exit 0
  fi

  echo "[$now] session=${SESSION_NAME} failed on attempt ${attempt}/${MAX_ATTEMPTS} exit_code=${cmd_rc}" | tee -a "$LOG_FILE"
  if (( attempt >= MAX_ATTEMPTS )); then
    break
  fi

  sleep "$SLEEP_SECONDS"
  attempt=$((attempt + 1))
done

now="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[$now] session=${SESSION_NAME} exhausted retries (${MAX_ATTEMPTS})" | tee -a "$LOG_FILE"
exit 1

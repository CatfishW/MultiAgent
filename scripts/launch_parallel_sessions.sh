#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="configs/system.experiments.yaml"
SPLIT="test"
LIMIT=""
OUT_ROOT="artifacts/experiments"
RESULTS_DIR="artifacts/experiments/results"
INDEX_ROOT="artifacts/experiments/indexes"
LOG_DIR="logs/experiments/sessions"
PROGRESS_EVERY="10"
SESSION_RETRIES="100"
SESSION_RETRY_SLEEP="20"
EXAMPLE_RETRIES="6"
EXAMPLE_5XX_RETRIES="2"
EXAMPLE_RETRY_BASE="2.0"
EXAMPLE_RETRY_MAX="45.0"
CHECKPOINT_EVERY="25"
EXAMPLE_TIMEOUT="300"
RESUME_MODE="1"
RESUME_ON_RETRY="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --out-dir)
      OUT_ROOT="$2"
      shift 2
      ;;
    --out-root)
      OUT_ROOT="$2"
      shift 2
      ;;
    --results-dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --index-root)
      INDEX_ROOT="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --progress-every)
      PROGRESS_EVERY="$2"
      shift 2
      ;;
    --session-retries)
      SESSION_RETRIES="$2"
      shift 2
      ;;
    --session-retry-sleep)
      SESSION_RETRY_SLEEP="$2"
      shift 2
      ;;
    --example-retries)
      EXAMPLE_RETRIES="$2"
      shift 2
      ;;
    --example-5xx-retries)
      EXAMPLE_5XX_RETRIES="$2"
      shift 2
      ;;
    --example-retry-base)
      EXAMPLE_RETRY_BASE="$2"
      shift 2
      ;;
    --example-retry-max)
      EXAMPLE_RETRY_MAX="$2"
      shift 2
      ;;
    --checkpoint-every)
      CHECKPOINT_EVERY="$2"
      shift 2
      ;;
    --example-timeout)
      EXAMPLE_TIMEOUT="$2"
      shift 2
      ;;
    --no-resume)
      RESUME_MODE="0"
      shift
      ;;
    --resume)
      RESUME_MODE="1"
      shift
      ;;
    --no-resume-on-retry)
      RESUME_ON_RETRY="0"
      shift
      ;;
    --resume-on-retry)
      RESUME_ON_RETRY="1"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$RESULTS_DIR" == "artifacts/experiments/results" ]]; then
  RESULTS_DIR="$OUT_ROOT/results"
fi
if [[ "$INDEX_ROOT" == "artifacts/experiments/indexes" ]]; then
  INDEX_ROOT="$OUT_ROOT/indexes"
fi

mkdir -p "$RESULTS_DIR" "$INDEX_ROOT" "$LOG_DIR"

RUNNER_SCRIPT="scripts/run_session_with_retry.sh"
if [[ ! -f "$RUNNER_SCRIPT" ]]; then
  echo "Missing runner script: $RUNNER_SCRIPT" >&2
  exit 1
fi

if ! command -v screen >/dev/null 2>&1; then
  echo "screen is not installed. Install it first (e.g., sudo apt install screen)." >&2
  exit 1
fi

EDUBENCH_SOURCE_OVERRIDE="${EDUBENCH_SOURCE_OVERRIDE:-}"
TUTOREVAL_SOURCE_OVERRIDE="${TUTOREVAL_SOURCE_OVERRIDE:-}"

declare -A DATASET_SOURCES=(
  [EduBench]="${EDUBENCH_SOURCE_OVERRIDE:-data/processed/edubench/${SPLIT}.jsonl}"
  [TutorEval]="${TUTOREVAL_SOURCE_OVERRIDE:-data/processed/tutoreval/${SPLIT}.jsonl}"
)

declare -A DATASET_CORPUS=(
  [EduBench]="data/processed/edubench/corpus.jsonl"
  [TutorEval]="data/processed/tutoreval/corpus.jsonl"
)

declare -A LEGACY_DATASET_SOURCES=(
  [EduBench]="data/edubench/${SPLIT}.jsonl"
  [TutorEval]="data/tutoreval/${SPLIT}.jsonl"
)

declare -A LEGACY_DATASET_CORPUS=(
  [EduBench]="data/edubench/corpus.jsonl"
  [TutorEval]="data/tutoreval/corpus.jsonl"
)

ARCHS=(
  "hybrid_fast"
  "classical_rag"
  "non_rag_multi_agent"
  "single_agent_no_rag"
)

ONLY_DATASETS="${ONLY_DATASETS:-}"  # space-separated list to restrict; empty = all

_should_run_dataset() {
  if [[ -z "$ONLY_DATASETS" ]]; then return 0; fi
  local target="$1"
  for d in $ONLY_DATASETS; do
    if [[ "$d" == "$target" ]]; then return 0; fi
  done
  return 1
}

for dataset in "${!DATASET_SOURCES[@]}"; do
  if ! _should_run_dataset "$dataset"; then
    echo "Skipping dataset (not in ONLY_DATASETS): $dataset"
    continue
  fi
  source_path="${DATASET_SOURCES[$dataset]}"
  corpus_path="${DATASET_CORPUS[$dataset]}"
  if [[ ! -f "$source_path" && -f "${LEGACY_DATASET_SOURCES[$dataset]}" ]]; then
    source_path="${LEGACY_DATASET_SOURCES[$dataset]}"
  fi
  if [[ ! -f "$corpus_path" && -f "${LEGACY_DATASET_CORPUS[$dataset]}" ]]; then
    corpus_path="${LEGACY_DATASET_CORPUS[$dataset]}"
  fi

  index_dir="$INDEX_ROOT/${dataset,,}"
  index_file="$index_dir/hybrid_index.pkl"

  if [[ ! -f "$source_path" ]]; then
    echo "Missing source file: $source_path" >&2
    exit 1
  fi

  if [[ -f "$corpus_path" && ! -f "$index_file" ]]; then
    echo "Building shared index for $dataset..."
    .venv/bin/python scripts/build_index.py "$corpus_path" --config "$CONFIG" --out "$index_dir"
  fi

  for arch in "${ARCHS[@]}"; do
    session="exp_${dataset,,}_${arch}"
    out_json="$RESULTS_DIR/${dataset,,}_${arch}.json"
    log_file="$LOG_DIR/${session}.log"

    extra_args=(
      "$dataset"
      "--config" "$CONFIG"
      "--source" "$source_path"
      "--split" "$SPLIT"
      "--architecture" "$arch"
      "--progress-every" "$PROGRESS_EVERY"
      "--max-example-retries" "$EXAMPLE_RETRIES"
      "--max-5xx-retries" "$EXAMPLE_5XX_RETRIES"
      "--retry-backoff-base" "$EXAMPLE_RETRY_BASE"
      "--retry-backoff-max" "$EXAMPLE_RETRY_MAX"
      "--checkpoint-every" "$CHECKPOINT_EVERY"
      "--example-timeout" "$EXAMPLE_TIMEOUT"
      "--allow-partial"
      "--out" "$out_json"
    )

    if [[ "$RESUME_MODE" == "1" ]]; then
      extra_args+=("--resume")
    else
      extra_args+=("--no-resume")
    fi

    if [[ -n "$LIMIT" ]]; then
      extra_args+=("--limit" "$LIMIT")
    fi

    if [[ "$arch" == "hybrid_fast" || "$arch" == "classical_rag" ]]; then
      if [[ -f "$index_file" ]]; then
        extra_args+=("--index-path" "$index_file")
      else
        echo "Missing index file for retrieval architecture: $index_file" >&2
        exit 1
      fi
    fi

    run_cmd=(
      ".venv/bin/python"
      "scripts/run_eval_session.py"
      "${extra_args[@]}"
    )

    wrapper_cmd=(
      "bash"
      "$RUNNER_SCRIPT"
      "--session-name" "$session"
      "--log-file" "$log_file"
      "--max-attempts" "$SESSION_RETRIES"
      "--sleep-seconds" "$SESSION_RETRY_SLEEP"
    )
    if [[ "$RESUME_ON_RETRY" == "1" ]]; then
      wrapper_cmd+=("--resume-on-retry")
    fi
    wrapper_cmd+=("--")
    wrapper_cmd+=("${run_cmd[@]}")
    printf -v wrapper_cmd_quoted '%q ' "${wrapper_cmd[@]}"
    # Inject safety env vars to reduce native crashes (segfaults, malloc corruption)
    # from tokenizer/threading race conditions.
    safety_env="export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false MALLOC_ARENA_MAX=2 PYTHONFAULTHANDLER=1;"
    cmd="cd '$ROOT_DIR' && ${safety_env} set -o pipefail && ${wrapper_cmd_quoted}"

    if screen -list | grep -q "\\.${session}[[:space:]]"; then
      echo "Session already exists, skipping: $session"
      continue
    fi

    screen -dmS "$session" bash -lc "$cmd"
    echo "Started $session"
    echo "  log: $log_file"
    echo "  out: $out_json"
  done
done

echo "All sessions launched."
echo "Check with: screen -list"
echo "Attach with: screen -r <session_name>"

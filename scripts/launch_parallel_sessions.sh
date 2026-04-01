#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="configs/system.experiments.yaml"
SPLIT="test"
LIMIT=""
OUT_DIR="artifacts/experiments"
LOG_DIR="logs/experiments"
PROGRESS_EVERY="10"

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
      OUT_DIR="$2"
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
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$OUT_DIR" "$LOG_DIR"

if ! command -v screen >/dev/null 2>&1; then
  echo "screen is not installed. Install it first (e.g., sudo apt install screen)." >&2
  exit 1
fi

declare -A DATASET_SOURCES=(
  [EduBench]="data/edubench/${SPLIT}.jsonl"
  [TutorEval]="data/tutoreval/${SPLIT}.jsonl"
)

declare -A DATASET_CORPUS=(
  [EduBench]="data/edubench/corpus.jsonl"
  [TutorEval]="data/tutoreval/corpus.jsonl"
)

ARCHS=(
  "hybrid_fast"
  "classical_rag"
  "non_rag_multi_agent"
  "single_agent_no_rag"
)

for dataset in "${!DATASET_SOURCES[@]}"; do
  source_path="${DATASET_SOURCES[$dataset]}"
  corpus_path="${DATASET_CORPUS[$dataset]}"
  index_dir="$OUT_DIR/index_${dataset,,}"
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
    out_json="$OUT_DIR/${dataset,,}_${arch}.json"
    log_file="$LOG_DIR/${session}.log"

    extra_args=(
      "$dataset"
      "--config" "$CONFIG"
      "--source" "$source_path"
      "--split" "$SPLIT"
      "--architecture" "$arch"
      "--progress-every" "$PROGRESS_EVERY"
      "--out" "$out_json"
    )

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

    cmd="cd '$ROOT_DIR' && set -o pipefail && .venv/bin/python scripts/run_eval_session.py ${extra_args[*]} 2>&1 | tee '$log_file'"

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

# Conference-Grade Educational Multi-Agent Framework (Python)

This repository now contains two layers:

1. **`agent_swarm_port/`** — the uploaded Python port of the reference swarm
   coordination subsystem.
2. **`eduagentic/`** — a new research-facing educational multi-agent framework
   built on top of that substrate for the architecture in your PDF.

The added implementation is designed for **high cohesion, low coupling, and
fast execution** against **local OpenAI-compatible endpoints**:

- text LLM endpoint: `https://game.agaii.org/llm/v1`
- multimodal endpoint: `https://game.agaii.org/mllm/v1`
- model discovery: `GET /models`
- optional: mark text endpoint as vision-capable with `supports_vision: true`

## What is implemented

### Architecture families
- `classical_rag`
- `agentic_rag`
- `non_rag_multi_agent`
- `single_agent_no_rag`
- `hybrid_fast` (default proposed method)

### Core research features
- OpenAI-compatible local endpoint clients with `/models` discovery
- fast hybrid retriever (word TF-IDF + char TF-IDF + compact latent projection)
- lightweight trainable reranker
- lightweight task-regime router for architecture selection
- student-state tracker for tutoring adaptivity
- specialist agents:
  - planner
  - diagnoser
  - retriever
  - rubric agent
  - tutor
  - critic
- optional bridge to the uploaded swarm runtime for ablations / reviewer checks
- benchmark adapters for every dataset family named in the paper
- evaluation utilities separating correctness, grounding, pedagogy, adaptivity,
  retrieval recall, and latency

## Why the proposed path should be faster than a big always-on RAG stack

The speed-oriented idea is not "more agents everywhere". It is:

- **conditional retrieval** instead of always-on retrieval
- **cheap planning and learner-state extraction** instead of repeated large-model
  orchestration calls
- **small linear reranking/router heads** instead of heavyweight secondary models
- **MMR context packing** to reduce redundant tokens
- **fast in-process runtime** as the default path, with the reference swarm kept
  optional for controlled comparisons rather than forced into every request

That makes the framework a much stronger research implementation than naive
"agent = more model calls" designs.

## Repository structure

```text
src/
  agent_swarm_port/        # uploaded reference port kept intact
  eduagentic/
    app.py                 # top-level system facade
    config.py              # config model + loaders
    core/                  # shared contracts
    llm/                   # endpoint clients + model registry
    retrieval/             # corpus loading, indexing, reranking, packing
    ml/                    # router + learner-state tracking
    agents/                # specialist agents
    orchestration/         # pipeline families + runtimes
    datasets/              # benchmark adapters + registry
    evaluation/            # metrics + evaluator
configs/
  models.example.yaml
  system.example.yaml
  datasets.example.yaml
scripts/
  inspect_models.py
  build_index.py
  sync_hf_datasets.py
  run_benchmark.py
  run_eval_session.py
  build_dashboard_data.py
  run_dashboard_live.sh
  run_web_server.sh
  train_router.py
  train_reranker.py
  demo_local_endpoints.py
data/
  raw/
    hf_datasets/
  processed/
    edubench/
    tutoreval/
artifacts/
  dataset_audit/
  experiments/
    indexes/
    results/
logs/
  experiments/
    sessions/
docs/
  EDU_SWARM_ARCHITECTURE.md
  DATASET_SUPPORT.md
tests/
  existing swarm-port tests
  new eduagentic tests
```

## Quick start

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Inspect local models

```bash
python scripts/inspect_models.py --config configs/models.example.yaml
```

### 3. Build an index for your course / benchmark corpus

```bash
python scripts/build_index.py ./data/course_docs --config configs/system.example.yaml --out artifacts/index
```

### 4. Run a quick question

```bash
python scripts/demo_local_endpoints.py "Explain photosynthesis with evidence." --config configs/system.example.yaml
```

### 5. Run a benchmark

```bash
python scripts/run_benchmark.py ScienceQA --config configs/system.example.yaml --limit 20 --out artifacts/scienceqa.json
```

### 6. Prepare EduBench / TutorEval local exports

```bash
python scripts/sync_hf_datasets.py --datasets EduBench TutorEval
python scripts/prepare_datasets.py --config configs/system.example.yaml --datasets EduBench TutorEval --split test --out-dir data/processed
```

The sync step verifies full upstream downloads and writes an audit report to `artifacts/dataset_audit/dataset_integrity_report.json`.

### 7. Launch parallel comparison sessions (screen)

```bash
./scripts/launch_parallel_sessions.sh --config configs/system.experiments.yaml --split test
```

This launches four architectures for each dataset:

- `hybrid_fast`
- `classical_rag`
- `non_rag_multi_agent`
- `single_agent_no_rag`

Logs are written to `logs/experiments/sessions/` and results to `artifacts/experiments/results/`.

Session retries now resume from the latest checkpoint by default, so failed attempts
continue unfinished examples instead of restarting from zero.
Dashboard progress counts processed examples (successes + terminal failures), so
long 5xx streaks do not look frozen; by default, 5xx retries are capped at `2`
per example.

## EduBench / TutorEval evaluation policy

The two datasets expose different supervision signals, so evaluation uses
dataset-aware mapping:

- `EduBench`
  - public rows typically do not ship a direct gold answer.
  - the adapter extracts consensus reference signals from `metadata.model_predictions`.
  - outputs are aligned to the EduBench 12-dimension rubric:
    - Scenario Adaptation: `edubench_iftc`, `edubench_rtc`, `edubench_crsc`, `edubench_sei`
    - Factual & Reasoning Accuracy: `edubench_bfa`, `edubench_dka`, `edubench_rpr`, `edubench_eicp`
    - Pedagogical Application: `edubench_csi`, `edubench_mgp`, `edubench_pas`, `edubench_hots`
  - grouped summaries are reported as:
    - `edubench_scenario_adaptation`
    - `edubench_factual_reasoning_accuracy`
    - `edubench_pedagogical_application`
    - `edubench_12d_mean`
- `TutorEval`
  - public rows use `metadata.key_points` as supervision.
  - key points are normalized into rubric items and a synthesized gold target.
  - primary metric is `tutoreval_keypoint_hit_rate`.
  - secondary metrics are `tutoreval_correctness`, `tutoreval_completeness`, and `tutoreval_relevance`.

This avoids the incorrect all-zero correctness metrics caused by missing direct
`answer` fields in the raw public exports.

### 8. Build dashboard data for the web monitor

```bash
python scripts/build_dashboard_data.py --results-dir artifacts/experiments/results --logs-dir logs/experiments/sessions --out web/data/session_summary.json
```

### 9. Run live dashboard refresh + web server (screen)

```bash
screen -dmS exp_dashboard_live ./scripts/run_dashboard_live.sh --interval 1
screen -dmS exp_web_server ./scripts/run_web_server.sh --port 8080
```

`run_web_server.sh` now serves the dashboard plus a websocket endpoint at `/ws`
that pushes `session_summary` updates every second.

## Dataset coverage

The registry includes support paths for all benchmark families named in the PDF:

### Education-specific
- EduBench
- TutorEval
- LM-Science-Tutor
- ScienceQA
- MathTutorBench
- TutorBench

### Transferable
- HotpotQA
- AgentBench
- SCROLLS
- LongBench-v2
- BEIR
- FEVER
- Wizard of Wikipedia

Some are configured with public Hugging Face defaults, while the more
repo-backed or environment-heavy ones expose local normalized adapters by
default. See `docs/DATASET_SUPPORT.md`.

## Notes on claims and evaluation

This codebase is designed to make your hypothesis *testable*. It does **not**
claim guaranteed empirical superiority before running the experiments. The right
conference-safe statement is:

- the implementation includes strong baselines,
- a faster conditional-retrieval hybrid,
- lightweight trainable routing/reranking modules,
- and a fair comparison harness for the architectural families in the paper.

## Run tests

```bash
python -m pytest
```

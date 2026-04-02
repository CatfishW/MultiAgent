# MultiAgent (Educational Multi-Agent Framework)

A research framework for evaluating educational QA/tutoring pipelines with multiple architectures, lightweight routing/reranking, and local OpenAI-compatible endpoints.

## What this repo contains

- `src/eduagentic/`: main framework (pipelines, retrieval, agents, datasets, evaluation)
- `src/agent_swarm_port/`: reference swarm runtime port kept for comparison/ablations
- `scripts/`: CLI tools for model inspection, indexing, benchmarking, training, experiments, dashboard data
- `configs/`: example configs for system/models/datasets
- `docs/`: architecture and dataset support notes

## Core framework

### Architecture families

- `hybrid_fast` (default): conditional retrieval + fallback retrieval when needed
- `classical_rag`: retrieve -> answer
- `agentic_rag`: planner/diagnoser/rubric + retrieval + answer
- `non_rag_multi_agent`: planner/diagnoser/rubric without retrieval
- `single_agent_no_rag`: single-agent baseline

### Main components

- **LLM layer**: local OpenAI-compatible endpoints with `/models` discovery
- **Retrieval layer**: hybrid index + reranking + MMR context packing
- **ML layer**: lightweight router and reranker (trainable)
- **Agent layer**: planner, diagnoser, retriever, tutor, rubric, critic
- **Orchestration layer**: fast in-process runtime + optional swarm adapter
- **Evaluation layer**: correctness/grounding/pedagogy/adaptivity/retrieval/latency metrics

## Core ideas (why this is fast)

1. Conditional retrieval instead of always-on retrieval
2. Cheap routing and student-state extraction
3. Small trainable heads (router/reranker) instead of heavy model stacks
4. In-process execution by default; swarm runtime remains optional for controlled comparisons

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional benchmark/test dependencies:

```bash
pip install -e ".[benchmarks,test]"
```

## Usage

### 1) Inspect available models

```bash
python scripts/inspect_models.py --config configs/models.example.yaml
```

### 2) Build retrieval index

```bash
python scripts/build_index.py ./data/course_docs --config configs/system.example.yaml --out artifacts/index
```

### 3) Quick question demo

```bash
python scripts/demo_local_endpoints.py "Explain photosynthesis with evidence." --config configs/system.example.yaml
```

### 4) Run benchmark

```bash
python scripts/run_benchmark.py ScienceQA --config configs/system.example.yaml --split test --limit 20 --out artifacts/benchmark_results.json
```

### 5) Prepare local dataset exports

```bash
python scripts/prepare_datasets.py --config configs/system.example.yaml --datasets EduBench TutorEval --split test --out-dir data
```

### 6) Run logged evaluation session

```bash
python scripts/run_eval_session.py ScienceQA --config configs/system.example.yaml --split test --limit 50 --out artifacts/eval_session_scienceqa.json
```

### 7) Launch parallel architecture experiments

```bash
./scripts/launch_parallel_sessions.sh --config configs/system.experiments.yaml --split test --limit 40
```

Outputs:
- Logs: `logs/experiments/`
- Results: `artifacts/experiments/`

### 8) Build dashboard summary

```bash
python scripts/build_dashboard_data.py --results-dir artifacts/experiments --logs-dir logs/experiments --out web/data/session_summary.json
```

## Training lightweight modules

Train router:

```bash
python scripts/train_router.py <training.jsonl> --config configs/system.example.yaml --out artifacts/router.pkl
```

Train reranker:

```bash
python scripts/train_reranker.py <training_pairs.jsonl> --out artifacts/reranker.pkl
```

## Experiment details

- `hybrid_fast` is the proposed method
- `classical_rag`, `agentic_rag`, `non_rag_multi_agent`, and `single_agent_no_rag` are comparison baselines
- Use fixed local dataset exports (JSONL/JSON) for reproducibility
- Prefer pinned config files in `configs/` and save outputs under `artifacts/`

For deeper details:
- `docs/EDU_SWARM_ARCHITECTURE.md`
- `docs/DATASET_SUPPORT.md`

## Tests

```bash
python -m pytest
```

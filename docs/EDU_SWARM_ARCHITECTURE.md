# Educational Multi-Agent Architecture

## Why this extension exists

This repository operationalizes the paper *Multi-Agent Retrieval as an
Alternative to Classical RAG: A Controlled Comparison on Educational
Tutoring Benchmarks* (MultiAgentOL/main_smc.tex). The central systems
question is whether a multi-agent coordination stack can **replace**
classical RAG's retrieve-then-read pipeline as the mechanism that supplies
grounded evidence to a tutoring model. The four pipelines below are the
four retrieval mechanisms compared in the paper; the code enum names map
one-to-one to the paper labels.

The uploaded Python port provided a swarm / mailbox / task-list substrate.
What it did **not** provide was the conference-facing layer needed to run
that comparison:

- local endpoint model discovery via `/models`
- fast classical RAG baseline and an agentic RAG variant
- multi-agent retrieval with a per-example retrieval gate
- tutoring / rubric / planning / diagnosis specialists
- benchmark adapters for every dataset family in the paper
- evaluation utilities that separate correctness, grounding, pedagogy, and cost

This project adds those missing layers without entangling them with the
reference runtime.

## High-level design

### 1. Fast path vs swarm path

The default path is `FastGraphRuntime`, which keeps independent specialists
in-process and avoids the file-backed overhead of the reference swarm when the
request is a normal benchmark example.

The optional `SwarmRuntimeAdapter` bridges selected specialist execution onto
`agent_swarm_port` for ablations or reviewer-required comparisons.

### 2. Four retrieval mechanisms (the paper's comparison axis)

Each code enum in `ArchitectureFamily` maps to one paper-level label.
All four mechanisms share the same tutor/critic backbone, the same corpus,
the same reranker, and the same per-call response budget; they differ only
in how evidence is supplied to the reader. In the released paper-grade runs,
the planner, diagnoser, and rubric stages are deterministic lightweight
modules, so `agent_count` measures materialized pipeline roles while
`llm_call_count` isolates actual backbone invocations.

| Code enum                  | Paper label                   | What it does                                                                                      |
| -------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------- |
| `CLASSICAL_RAG`            | *Classical RAG* (baseline)    | retrieve -> pack context -> answer -> optional critic; retriever is the top-level controller     |
| `NON_RAG_MULTI_AGENT`      | *Multi-Agent (no retrieval)*  | planner \| diagnoser \| rubric -> answer -> critic; no retriever                                  |
| `SINGLE_AGENT_NO_RAG`      | *Single-Agent (no retrieval)* | tutor -> optional critic; ablation floor                                                          |
| `HYBRID_FAST`              | *Multi-Agent Retrieval* (ours) | planner \| diagnoser \| rubric -> gate -> retriever (if gate fires) -> tutor -> critic -> fallback |
| `AGENTIC_RAG`              | *(not reported in the paper)* | planner \| diagnoser \| rubric -> retrieve-always -> tutor -> critic; parity-controlled outputs unavailable |

The paper reports only the four rows with comparable outputs; `AGENTIC_RAG`
remains implemented for sanity-checking but is excluded from the headline
table.

### 3. Lightweight ML modules

The implementation deliberately uses *small* trainable components instead of
heavy model-over-model orchestration:

- `LightweightRegimeRouter`: keyword-aware + optional logistic regression router
- `LightweightReranker`: optional logistic regression reranker over cheap pair
  features
- `StudentStateTracker`: rule-based learner-state extraction that avoids
  repeatedly spending model calls on the same metadata

These modules are inexpensive enough to run inside the orchestration loop,
which is one of the main speed advantages over large always-on RAG stacks.

## Package map

- `src/eduagentic/llm/` — OpenAI-compatible local endpoint clients and model registry
- `src/eduagentic/retrieval/` — chunking, hybrid indexing, reranking, packing
- `src/eduagentic/ml/` — lightweight routers and student state modeling
- `src/eduagentic/agents/` — planner, diagnoser, retriever, tutor, rubric, critic
- `src/eduagentic/orchestration/` — fast runtime, swarm bridge, pipeline families
- `src/eduagentic/datasets/` — adapters and registry for benchmark families
- `src/eduagentic/evaluation/` — metrics and benchmark evaluator
- `src/agent_swarm_port/` — uploaded reference swarm substrate kept intact

## Speed-oriented decisions

1. Retrieval is conditional in the hybrid path, not always-on.
2. Planning and student-state extraction are cheap by default.
3. The retriever uses sparse + char + compact latent projections instead of a
   heavyweight embedding service.
4. The reranker is a tiny linear head over engineered features.
5. Context packing uses MMR to reduce redundant tokens.
6. The reference swarm runtime is optional so most benchmark runs do not pay
   file-backed coordination overhead.

## Paper-grade audit trail

Use `scripts/audit_paper_runs.py` to regenerate the metadata that the paper
quotes from archived run files: model identifiers, thinking budget, corpus
row counts, index chunk counts, retrieval rates, `agent_count`,
`llm_call_count`, `complexity_units`, and `corpus_factuality`. This avoids
drifting from the released result JSONs when tables or prose are updated.

## Expected research use

Use `hybrid_fast` as the main proposed method, but keep `classical_rag`,
`agentic_rag`, and `non_rag_multi_agent` available for fairness controls and
ablations.

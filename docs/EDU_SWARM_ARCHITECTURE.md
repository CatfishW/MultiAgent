# Educational Multi-Agent Architecture

## Why this extension exists

The uploaded Python port already provided a faithful swarm / mailbox / task-list
substrate. What it did **not** provide was the conference-facing layer needed
for educational multi-agent research:

- local endpoint model discovery via `/models`
- fast classical RAG and agentic RAG baselines
- conditional-retrieval hybrid orchestration
- tutoring / rubric / planning specialists
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

### 2. Four architecture families

- `classical_rag`: retrieve -> pack context -> answer -> optional critic
- `agentic_rag`: planner/diagnoser/rubric -> retrieve -> answer -> critic
- `non_rag_multi_agent`: planner/diagnoser/rubric -> answer -> critic
- `hybrid_fast`: conditionally retrieve only when the task actually needs it,
  with a fallback retrieval pass if the draft answer signals under-grounding

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

## Expected research use

Use `hybrid_fast` as the main proposed method, but keep `classical_rag`,
`agentic_rag`, and `non_rag_multi_agent` available for fairness controls and
ablations.

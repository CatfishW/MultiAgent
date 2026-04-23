# Reviewer-Critical Experiments

This index is the single source of truth for every experiment we added to
respond to the reviewer feedback summarized in the paper repository
(`REVIEWER_FEEDBACK.md`). Each entry has the reviewer concern, the exact
command to run, where outputs land, and the corresponding slot in the paper
(`MultiAgentOL/main_smc.tex`).

All commands assume:
- Python environment at `.venv` with `pip install -e .` already executed.
- Qwen-compatible endpoint defined in `configs/system.experiments.yaml`.
- Shared corpus index prebuilt under `artifacts/experiments/indexes/<dataset>/`.

Unless noted otherwise, every runner reuses `scripts/run_eval_session.py` under
the hood so resume/backoff semantics stay identical to the production runs.

## 1. Paired statistical significance (bootstrap CI + permutation test)

Reviewer concern: no confidence intervals or p-values on TutorEval deltas.

```bash
python scripts/compute_paired_stats.py \
  --session hybrid=artifacts/experiments/results/tutoreval_hybrid_fast.json \
  --session classical=artifacts/experiments/results/tutoreval_classical_rag.json \
  --session non_rag=artifacts/experiments/results/tutoreval_non_rag_multi_agent.json \
  --session single=artifacts/experiments/results/tutoreval_single_agent_no_rag.json \
  --compare classical,hybrid \
  --metric token_f1 --metric tutoreval_keypoint_hit_rate --metric rubric_coverage \
          --metric latency_ms --metric total_tokens --metric corpus_factuality \
  --bootstrap 2000 --permutations 10000 --confidence 0.95 \
  --label tutoreval_main --out-dir artifacts/stats
```

Outputs: `artifacts/stats/tutoreval_main/{pairs.tsv,pairs.jsonl,inputs.json}`.

Paper slot: *Experimental Results / Updated quality-efficiency comparison*.

## 2. Ablations

Reviewer concern: cannot attribute the hybrid's win to any specific mechanism.

All ablations use the same dataset, index, architecture, and Qwen endpoint as
the baseline run; only the config flags differ.

### A. Hybrid without conditional retrieval

```bash
python scripts/run_ablation_session.py TutorEval \
  --config configs/system.experiments.yaml \
  --architecture hybrid_fast --ablation hybrid_force_retrieval \
  --source data/processed/tutoreval/test.jsonl \
  --index-path artifacts/experiments/indexes/tutoreval/hybrid_index.pkl \
  --out-dir artifacts/ablations
```

### B. Non-RAG with retrieval enabled

```bash
python scripts/run_ablation_session.py TutorEval \
  --config configs/system.experiments.yaml \
  --architecture non_rag_multi_agent --ablation non_rag_enable_retrieval \
  --index-path artifacts/experiments/indexes/tutoreval/hybrid_index.pkl
```

### C. Hybrid without critic

```bash
python scripts/run_ablation_session.py TutorEval \
  --config configs/system.experiments.yaml \
  --architecture hybrid_fast --ablation hybrid_disable_critic \
  --index-path artifacts/experiments/indexes/tutoreval/hybrid_index.pkl
```

### D. Router: heuristic-only vs classifier-only

```bash
python scripts/run_ablation_session.py EduBench \
  --config configs/system.experiments.yaml \
  --architecture hybrid_fast --ablation router_heuristic_only
python scripts/run_ablation_session.py EduBench \
  --config configs/system.experiments.yaml \
  --architecture hybrid_fast --ablation router_classifier_only
```

Outputs: `artifacts/ablations/<ablation_tag>/<dataset>_<arch>.json`. Each file
also stores the materialized config under `configs/ablation_<tag>.yaml`.

Paper slot: *Experimental Results / Planned controlled experiments*.

## 3. Router threshold sensitivity sweep

Reviewer concern: conditional retrieval sits on a single pair of thresholds.

```bash
python scripts/run_threshold_sweep.py TutorEval \
  --config configs/system.experiments.yaml \
  --architecture hybrid_fast \
  --thresholds 0.30 0.35 0.40 0.45 0.50 0.55 \
  --index-path artifacts/experiments/indexes/tutoreval/hybrid_index.pkl \
  --limit 200 --label tau_e_main
```

Outputs: `artifacts/sweeps/tau_e_main/{threshold_sweep.tsv, <dataset>_<arch>_tauE*_gate*.json}`.

Paper slot: *Methodology / Regime-aware routing* (sensitivity sentence) and
*Planned controlled experiments*.

## 4. Retrieval-agnostic grounding metric

Reviewer concern: `grounded_overlap` structurally returns `0.0` for non-RAG
architectures. `corpus_factuality` scores every architecture against the same
shared corpus index regardless of whether retrieval was invoked.

- Implementation: `src/eduagentic/evaluation/metrics.py::corpus_factuality`
- Automatically populated by `scripts/run_eval_session.py` whenever an
  `--index-path` or `--corpus` is supplied.
- Baseline reruns therefore also produce this metric under the same summary
  key (`corpus_factuality`).

Paper slot: *Evaluation protocol* paragraph.

## 5. Novelty vs Self-RAG / CRAG

Reviewer concern: the proposed conditional retrieval is not clearly
differentiated from Self-RAG or CRAG.

- The paper side contains the comparison table (`MultiAgentOL/main_smc.tex`,
  Self-RAG/CRAG positioning table under Related Work).
- No code change required beyond this doc; the table is positioning-only and
  does not claim new numeric results.

## 6. Human-evaluation pack

Reviewer concern: no human evaluation of tutoring quality.

```bash
python scripts/export_human_eval_pack.py \
  --session hybrid=artifacts/experiments/results/tutoreval_hybrid_fast.json \
  --session classical=artifacts/experiments/results/tutoreval_classical_rag.json \
  --session non_rag=artifacts/experiments/results/tutoreval_non_rag_multi_agent.json \
  --session single=artifacts/experiments/results/tutoreval_single_agent_no_rag.json \
  --per-arch 25 --raters 3 --out-dir artifacts/human_eval/tutoreval

# After raters fill rubric.csv:
python scripts/export_human_eval_pack.py --iaa artifacts/human_eval/tutoreval/rubric.csv
```

Outputs: `artifacts/human_eval/tutoreval/{items.jsonl, rubric.csv, protocol.md, manifest.json}`.

Paper slot: *Limitations* (human-eval protocol pointer) and planned
camera-ready extension.

## 7. Transfer-benchmark micro-study (optional)

Reviewer concern: transfer benchmarks mentioned but not used.

The dataset registry already supports `HotpotQA`, `FEVER`, `BEIR`, `LongBench2`,
`SCROLLS`, `AgentBench`, and `WizardOfWikipedia`. To generate a short transfer
comparison (reviewer-safe, mechanism-focused), run:

```bash
for arch in hybrid_fast classical_rag non_rag_multi_agent single_agent_no_rag; do
  python scripts/run_ablation_session.py HotpotQA \
    --config configs/system.experiments.yaml \
    --architecture "$arch" --ablation baseline \
    --source data/processed/hotpotqa/test.jsonl \
    --index-path artifacts/experiments/indexes/hotpotqa/hybrid_index.pkl \
    --limit 200 --out-dir artifacts/transfer
done
```

Paper slot: *Related Work / RAG and agentic retrieval control* (mechanism
claim) and *Planned controlled experiments*.

## Conventions

- All JSON outputs include per-example `ablation.*` telemetry keys coming from
  `BasePipeline._ablation_metrics`, so downstream tools can stratify without
  re-parsing config files.
- Every new script exits non-zero if the Qwen endpoint, dataset file, or index
  is missing, so silent empty runs are prevented.
- Tests under `tests/test_pipeline_ablations.py` cover the critical ablation
  paths with FakeChatClient to guarantee that default behavior is preserved
  when flags are unset.

# latex_tables_metrics_

This folder contains an IEEE SMC 2026 style result packet built from the latest available experiment artifacts at snapshot time.

Snapshot provenance:
- Source files: artifacts/experiments/results/*.json
- Freeze timestamp: 2026-04-12 21:18:44 EDT
- Frozen raw snapshot: frozen_metrics_snapshot.jsonl

Contents:
- smc2026_full_section.tex: standalone IEEE conference document wrapper.
- smc2026_narrative_and_methodology.tex: conference-style narrative with full metric semantics and novel architecture proposal.
- smc2026_tables.tex: primary and exhaustive metric inventory tables for EduBench and TutorEval.
- smc2026_figures.tex: PGFPlots/TikZ figures for quality, efficiency, completion, and proposed controller design.
- frozen_metrics_snapshot.jsonl: machine-readable frozen metrics and run status.
- frozen_metrics_snapshot.tsv: compact tabular extraction from the frozen snapshot.
- frozen_metrics_derived.tsv: derived coverage and failure rates plus key metrics.

Compile example:
1. cd latex_tables_metrics_
2. pdflatex smc2026_full_section.tex
3. pdflatex smc2026_full_section.tex

Notes for interpretation:
- EduBench runs are in-progress in this snapshot; metrics are partial and may evolve.
- TutorEval runs are fully processed in this snapshot.
- Profile-specific metrics are intentionally zero outside their dataset profile (EduBench vs TutorEval).
- API-time and complexity-per-second summary means should be interpreted with the caveats documented in smc2026_narrative_and_methodology.tex.
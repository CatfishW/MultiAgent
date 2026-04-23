#!/usr/bin/env python3
"""Sweep router thresholds for the hybrid pipeline and collect summary metrics.

Reviewer ask: sensitivity of the conditional retrieval mechanism to the
evidence-score threshold ``tau_e`` (and the hybrid retrieval gate). This script
generates a scratch config per threshold setting, runs an evaluation session
via ``run_eval_session``, and aggregates the resulting metric summaries into a
single TSV for easy plotting.

Defaults target the hybrid_fast pipeline on a TutorEval-like dataset with a
small ``--limit`` so the sweep is tractable in minutes; bump the limit for
production-scale sweeps.

Usage::

    python scripts/run_threshold_sweep.py TutorEval \
        --config configs/system.experiments.yaml \
        --architecture hybrid_fast \
        --thresholds 0.30 0.35 0.40 0.45 0.50 \
        --limit 120

Outputs (per run):
  artifacts/sweeps/<label>/<dataset>_<arch>_tauE_<value>.json
  artifacts/sweeps/<label>/threshold_sweep.tsv
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_eval_session  # noqa: E402


HEADLINE_KEYS: tuple[str, ...] = (
    "token_f1",
    "rubric_coverage",
    "grounded_overlap",
    "corpus_factuality",
    "edubench_12d_mean",
    "tutoreval_keypoint_hit_rate",
    "edu_score_alignment",
    "latency_ms",
    "total_tokens",
    "llm_call_count",
    "retrieval_query_count",
)


def _merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _write_config(base_path: str, evidence_threshold: float, gate: float, fallback: float, out_path: Path) -> None:
    base = yaml.safe_load(Path(base_path).read_text(encoding="utf-8")) or {}
    overrides = {
        "router": {
            "evidence_threshold": float(evidence_threshold),
            "hybrid_retrieval_gate": float(gate),
            "hybrid_retrieval_fallback": float(fallback),
        },
        "pipeline": {"ablation_tag": f"tauE={evidence_threshold:.2f}_gate={gate:.2f}"},
    }
    merged = _merge(base, overrides)
    out_path.write_text(yaml.safe_dump(merged, sort_keys=True), encoding="utf-8")


def _build_session_args(raw: argparse.Namespace, config_path: Path, out_path: Path) -> argparse.Namespace:
    return SimpleNamespace(
        dataset_name=raw.dataset_name,
        config=str(config_path),
        source=raw.source,
        split=raw.split,
        architecture=raw.architecture,
        limit=raw.limit,
        corpus=raw.corpus,
        index_path=raw.index_path,
        index_out=None,
        progress_every=raw.progress_every,
        resume=False,
        max_example_retries=raw.max_example_retries,
        max_5xx_retries=raw.max_5xx_retries,
        retry_backoff_base=raw.retry_backoff_base,
        retry_backoff_max=raw.retry_backoff_max,
        checkpoint_every=max(5, raw.checkpoint_every),
        allow_partial=True,
        example_timeout=raw.example_timeout,
        out=str(out_path),
    )


def _summary_row(payload: dict[str, Any], tau_e: float, gate: float, fallback: float) -> dict[str, Any]:
    summary = (payload.get("result") or {}).get("summary") or {}
    row: dict[str, Any] = {
        "evidence_threshold": f"{tau_e:.2f}",
        "hybrid_retrieval_gate": f"{gate:.2f}",
        "hybrid_retrieval_fallback": f"{fallback:.2f}",
        "records": int((payload.get("result") or {}).get("count", 0) or 0),
    }
    for key in HEADLINE_KEYS:
        value = summary.get(key)
        if isinstance(value, (int, float)):
            row[key] = float(value)
        else:
            row[key] = ""
    return row


def _write_tsv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    headers = [
        "evidence_threshold",
        "hybrid_retrieval_gate",
        "hybrid_retrieval_fallback",
        "records",
        *HEADLINE_KEYS,
    ]
    lines = ["\t".join(headers)]
    for row in rows:
        lines.append("\t".join(str(row.get(key, "")) for key in headers))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def _run_sweep(raw: argparse.Namespace, label: str) -> list[dict[str, Any]]:
    out_root = Path(raw.out_dir) / label
    out_root.mkdir(parents=True, exist_ok=True)
    config_dir = out_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for tau_e in raw.thresholds:
        gate = raw.gate if raw.gate is not None else tau_e
        fallback = raw.fallback if raw.fallback is not None else max(tau_e + 0.10, 0.45)
        safe_tag = f"tauE{tau_e:.2f}_gate{gate:.2f}".replace(".", "p")
        cfg_path = config_dir / f"config_{safe_tag}.yaml"
        _write_config(raw.config, tau_e, gate, fallback, cfg_path)
        result_path = out_root / f"{raw.dataset_name.lower()}_{raw.architecture}_{safe_tag}.json"
        session_args = _build_session_args(raw, cfg_path, result_path)
        started = time.time()
        payload = await run_eval_session._run(session_args)  # noqa: SLF001
        run_eval_session._checkpoint_write(result_path, payload)  # noqa: SLF001
        duration = round(time.time() - started, 3)
        row = _summary_row(payload, tau_e, gate, fallback)
        row["duration_s"] = duration
        row["out"] = str(result_path)
        rows.append(row)
        print(json.dumps({"step": safe_tag, "duration_s": duration, "out": str(result_path)}, ensure_ascii=False))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_name")
    parser.add_argument("--config", required=True)
    parser.add_argument("--architecture", default="hybrid_fast")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55])
    parser.add_argument("--gate", type=float, default=None, help="Override hybrid retrieval gate (defaults to each tau_e).")
    parser.add_argument("--fallback", type=float, default=None, help="Override hybrid fallback threshold.")
    parser.add_argument("--out-dir", default="artifacts/sweeps")
    parser.add_argument("--label", default=None, help="Sweep label subdirectory; defaults to timestamp.")
    parser.add_argument("--source", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--index-path", default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--max-example-retries", type=int, default=4)
    parser.add_argument("--max-5xx-retries", type=int, default=2)
    parser.add_argument("--retry-backoff-base", type=float, default=2.0)
    parser.add_argument("--retry-backoff-max", type=float, default=30.0)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--example-timeout", type=float, default=300.0)
    raw = parser.parse_args()

    label = raw.label or time.strftime("sweep_%Y%m%d_%H%M%S")
    rows = asyncio.run(_run_sweep(raw, label))
    tsv_path = Path(raw.out_dir) / label / "threshold_sweep.tsv"
    _write_tsv(rows, tsv_path)
    print(json.dumps({"summary_tsv": str(tsv_path), "steps": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

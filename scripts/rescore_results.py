#!/usr/bin/env python3
"""Re-score existing experiment result JSONs with the audited metrics.

Walks `artifacts/experiments/results/*.json`, reconstructs `BenchmarkExample`
shells from each record plus its dataset source file (for metadata access),
re-runs `compute_metrics` using the stored answer + retrieved_doc_ids, and
rewrites `summary` via the fixed `summarize` (union + NaN-skip).

Writes alongside the originals as `<name>.rescored.json` and produces a
`<name>.diff.json` with per-key summary deltas. `--promote` swaps the
rescored files into place and gzip-archives originals under
`artifacts/experiments/results_pre_audit/`.
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path
from typing import Any

from eduagentic.core.contracts import (
    ArchitectureFamily,
    BenchmarkExample,
    Modality,
    PipelineResponse,
    RetrievedChunk,
    RouteDecision,
    TaskRegime,
)
from eduagentic.evaluation.metrics import compute_metrics, summarize


def _dataset_source_for(meta: dict[str, Any]) -> Path | None:
    source = meta.get("source")
    if not source:
        return None
    p = Path(source)
    if p.exists():
        return p
    # fall back to canonical locations
    dataset = str(meta.get("dataset", "")).lower()
    candidates = [
        Path(f"data/processed/{dataset}/test.jsonl"),
        Path(f"data/{dataset}/test.jsonl"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _load_examples_by_id(path: Path) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            eid = str(row.get("id", ""))
            if eid:
                by_id[eid] = row
    return by_id


def _mk_example(row: dict[str, Any], dataset_name: str) -> BenchmarkExample:
    metadata = row.get("metadata") or {}
    return BenchmarkExample(
        example_id=str(row.get("id", "")),
        dataset_name=dataset_name,
        regime_hint=None,
        question=str(row.get("question", "")),
        gold_answer=row.get("answer"),
        choices=row.get("choices"),
        context_text=row.get("context"),
        dialogue_history=[],
        rubric=row.get("rubric"),
        images=row.get("images"),
        metadata=metadata if isinstance(metadata, dict) else {},
        reference_docs=row.get("reference_docs") or [],
        expected_doc_ids=row.get("expected_doc_ids") or [],
    )


def _mk_response(record: dict[str, Any]) -> PipelineResponse:
    answer = str(record.get("answer", ""))
    doc_ids = record.get("retrieved_doc_ids") or []
    chunks: list[RetrievedChunk] = []
    for i, did in enumerate(doc_ids):
        chunks.append(RetrievedChunk(chunk_id=f"rechunk_{i}", doc_id=str(did), title="", text="", score=0.0))
    route = RouteDecision(
        regime=TaskRegime.EVIDENCE_GROUNDED,
        architecture=ArchitectureFamily.HYBRID_FAST,
        require_retrieval=False,
        use_critic=False,
        use_rubric_agent=False,
        modality=Modality.TEXT,
    )
    # Preserve latency/token metrics already recorded so rescored summary keeps cost info.
    preserved = {
        k: record.get("metrics", {}).get(k)
        for k in [
            "latency_ms",
            "api_time_ms",
            "non_api_time_ms",
            "api_time_ratio",
            "agent_count",
            "llm_call_count",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "retrieval_query_count",
            "complexity_units",
            "complexity_per_second",
        ]
        if isinstance(record.get("metrics"), dict) and k in record["metrics"]
    }
    return PipelineResponse(
        answer=answer,
        architecture=ArchitectureFamily.HYBRID_FAST,
        regime=TaskRegime.EVIDENCE_GROUNDED,
        route=route,
        citations=doc_ids,
        retrieved_chunks=chunks,
        agent_outputs={},
        metrics=preserved,
        trace=[],
    )


def _rescore_file(path: Path) -> tuple[Path, Path, dict[str, Any]] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"  skip unreadable {path}: {exc}")
        return None
    meta = payload.get("meta") or {}
    result = payload.get("result") or {}
    records = result.get("records") or []
    dataset_name = str(meta.get("dataset", ""))
    src = _dataset_source_for(meta)
    examples_by_id: dict[str, dict[str, Any]] = {}
    if src is not None:
        examples_by_id = _load_examples_by_id(src)
    else:
        print(f"  warning: no dataset source for {path.name}; metadata-dependent metrics may be degraded")

    new_records: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        eid = str(rec.get("example_id", ""))
        row = examples_by_id.get(eid, {})
        ex = _mk_example(row or {"id": eid, "question": ""}, dataset_name=dataset_name)
        resp = _mk_response(rec)
        metrics = compute_metrics(ex, resp)
        # preserve success flag if present, else infer as True (record exists with answer)
        prev = rec.get("metrics") if isinstance(rec.get("metrics"), dict) else {}
        metrics["success"] = float(prev.get("success", 1.0))
        metrics["has_gold"] = 1.0 if ex.gold_answer else 0.0
        metrics["has_rubric"] = 1.0 if ex.rubric else 0.0
        ref_score = ex.metadata.get("edubench_reference_score_mean")
        metrics["has_reference_score"] = 1.0 if isinstance(ref_score, (int, float)) else 0.0
        new_rec = dict(rec)
        new_rec["metrics"] = metrics
        new_rec["success"] = True
        new_rec["evaluation_profile"] = str(ex.metadata.get("evaluation_profile", "generic"))
        new_records.append(new_rec)
        metric_rows.append(metrics)

    new_summary = summarize(metric_rows) if metric_rows else {}
    new_result = dict(result)
    new_result["records"] = new_records
    new_result["summary"] = new_summary
    new_payload = dict(payload)
    new_payload["result"] = new_result
    new_meta = dict(meta)
    new_meta["metrics_audit_version"] = "2026-04-20"
    new_payload["meta"] = new_meta

    rescored = path.with_suffix(".rescored.json")
    rescored.write_text(json.dumps(new_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    old_summary = result.get("summary") or {}
    diff: dict[str, Any] = {}
    for key in sorted(set(old_summary) | set(new_summary)):
        old_v = old_summary.get(key)
        new_v = new_summary.get(key)
        if isinstance(old_v, (int, float)) and isinstance(new_v, (int, float)):
            delta = float(new_v) - float(old_v)
            diff[key] = {"old": old_v, "new": new_v, "delta": delta}
        else:
            diff[key] = {"old": old_v, "new": new_v}
    diff_path = path.with_suffix(".diff.json")
    diff_path.write_text(json.dumps({"records": len(new_records), "summary_diff": diff}, ensure_ascii=False, indent=2), encoding="utf-8")
    return rescored, diff_path, diff


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="artifacts/experiments/results")
    ap.add_argument("--promote", action="store_true", help="After rescoring, swap canonical files and archive originals.")
    ap.add_argument("--archive-dir", default="artifacts/experiments/results_pre_audit")
    args = ap.parse_args()

    root = Path(args.results_dir)
    paths = sorted(p for p in root.glob("*.json") if not p.name.endswith(".rescored.json") and not p.name.endswith(".diff.json"))
    if not paths:
        print(f"no result files under {root}")
        return 1

    results: list[tuple[Path, Path, Path]] = []
    for path in paths:
        print(f"rescoring {path.name}...")
        out = _rescore_file(path)
        if out is None:
            continue
        rescored, diff_path, _ = out
        results.append((path, rescored, diff_path))
        print(f"  -> {rescored.name} / {diff_path.name}")

    if args.promote:
        archive = Path(args.archive_dir)
        archive.mkdir(parents=True, exist_ok=True)
        for original, rescored, _ in results:
            gz = archive / (original.name + ".gz")
            with original.open("rb") as fi, gzip.open(gz, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            original.unlink()
            rescored.rename(original)
            print(f"promoted {original.name} (archived to {gz})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

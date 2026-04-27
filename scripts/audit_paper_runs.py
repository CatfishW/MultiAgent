#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _count_jsonl_rows(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _infer_corpus_path(meta: dict[str, Any]) -> Path | None:
    source = meta.get("source")
    if not isinstance(source, str) or not source:
        return None
    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = (REPO_ROOT / source_path).resolve()
    sibling = source_path.with_name("corpus.jsonl")
    return sibling if sibling.exists() else None


def _infer_index_path(meta: dict[str, Any], run_path: Path) -> Path | None:
    index_path = meta.get("index_path")
    if not isinstance(index_path, str) or not index_path:
        return None
    path = Path(index_path)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path if path.exists() else None


def _load_index_stats(index_path: Path | None) -> dict[str, Any]:
    if index_path is None:
        return {"index_chunks": None, "index_error": None}
    try:
        with index_path.open("rb") as fh:
            index = pickle.load(fh)
        return {
            "index_chunks": len(getattr(index, "chunks", []) or []),
            "index_error": None,
        }
    except Exception as exc:  # pragma: no cover - best-effort audit tool
        return {
            "index_chunks": None,
            "index_error": f"{type(exc).__name__}: {exc}",
        }


def _thinking_budget(meta: dict[str, Any]) -> Any:
    return (
        meta.get("text_chat_extra", {})
        .get("extra_body", {})
        .get("thinking_budget")
    )


def _retrieval_rate(records: list[dict[str, Any]]) -> float | None:
    if not records:
        return None
    retrieved = sum(1 for record in records if record.get("retrieved_doc_ids"))
    return retrieved / len(records)


def _record_summary(run_path: Path) -> dict[str, Any]:
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    meta = payload["meta"]
    result = payload["result"]
    summary = result.get("summary", {})
    records = result.get("records", [])

    corpus_path = _infer_corpus_path(meta)
    index_path = _infer_index_path(meta, run_path)
    index_stats = _load_index_stats(index_path)

    row = {
        "run_path": str(run_path.resolve()),
        "dataset": meta.get("dataset"),
        "architecture": meta.get("architecture"),
        "config": meta.get("config"),
        "count": result.get("count"),
        "success_count": result.get("success_count"),
        "text_model": meta.get("text_model"),
        "vision_model": meta.get("vision_model"),
        "thinking_budget": _thinking_budget(meta),
        "llm_call_count": summary.get("llm_call_count"),
        "agent_count": summary.get("agent_count"),
        "retrieval_rate": _retrieval_rate(records),
        "retrieval_query_count": summary.get("retrieval_query_count"),
        "latency_ms": summary.get("latency_ms"),
        "total_tokens": summary.get("total_tokens"),
        "complexity_units": summary.get("complexity_units"),
        "corpus_factuality": summary.get("corpus_factuality"),
        "token_f1": summary.get("token_f1"),
        "rubric_coverage": summary.get("rubric_coverage"),
        "tutoreval_keypoint_hit_rate": summary.get("tutoreval_keypoint_hit_rate"),
        "edubench_12d_mean": summary.get("edubench_12d_mean"),
        "edu_score_alignment": summary.get("edu_score_alignment"),
        "corpus_path": str(corpus_path) if corpus_path else None,
        "corpus_docs": _count_jsonl_rows(corpus_path),
        "index_path": str(index_path) if index_path else None,
    }
    row.update(index_stats)
    return row


def _markdown(rows: list[dict[str, Any]]) -> str:
    columns = [
        "dataset",
        "architecture",
        "count",
        "thinking_budget",
        "retrieval_query_count",
        "llm_call_count",
        "agent_count",
        "retrieval_rate",
        "complexity_units",
        "latency_ms",
        "total_tokens",
        "corpus_factuality",
        "corpus_docs",
        "index_chunks",
        "text_model",
    ]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                values.append(f"{value:.4f}" if column == "retrieval_rate" else f"{value:.2f}")
            else:
                values.append("" if value is None else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit paper-grade run metadata from archived result JSONs.")
    parser.add_argument("--run", action="append", required=True, help="Path to a run JSON file. Repeat for multiple runs.")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    args = parser.parse_args()

    rows = [_record_summary((REPO_ROOT / run).resolve() if not Path(run).is_absolute() else Path(run)) for run in args.run]
    if args.format == "markdown":
        print(_markdown(rows))
    else:
        print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

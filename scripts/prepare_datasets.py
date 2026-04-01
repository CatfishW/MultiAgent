from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from eduagentic.app import ConferenceEduSystem
from eduagentic.core.contracts import BenchmarkExample
from eduagentic.datasets.adapters import generic_text_transform
from eduagentic.utils.text import stable_hash
from huggingface_hub import snapshot_download


def _slug(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def _example_to_row(example: BenchmarkExample) -> dict[str, Any]:
    return {
        "id": example.example_id,
        "question": example.question,
        "answer": example.gold_answer,
        "context": example.context_text,
        "dialogue": [asdict(turn) for turn in example.dialogue_history],
        "rubric": list(example.rubric) if example.rubric else None,
        "images": list(example.images) if example.images else None,
        "reference_docs": _to_jsonable(example.reference_docs),
        "expected_doc_ids": list(example.expected_doc_ids),
        "metadata": _to_jsonable(example.metadata),
    }


def _extract_corpus_rows(example: BenchmarkExample) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if example.context_text and example.context_text.strip():
        rows.append(
            {
                "id": f"{example.example_id}:context",
                "title": f"{example.dataset_name} context {example.example_id}",
                "text": example.context_text.strip(),
                "source_type": "context",
                "example_id": example.example_id,
                "dataset": example.dataset_name,
            }
        )

    for idx, doc in enumerate(example.reference_docs):
        if not isinstance(doc, dict):
            continue
        text = str(
            doc.get("text")
            or doc.get("content")
            or doc.get("passage")
            or doc.get("body")
            or doc.get("document")
            or ""
        ).strip()
        if not text:
            continue
        doc_id = str(doc.get("id") or doc.get("doc_id") or f"{example.example_id}:ref:{idx}")
        title = str(doc.get("title") or doc.get("name") or doc_id)
        rows.append(
            {
                "id": doc_id,
                "title": title,
                "text": text,
                "source_type": doc.get("source_type", "reference"),
                "example_id": example.example_id,
                "dataset": example.dataset_name,
            }
        )

    # Heuristic pass: include nested metadata strings as weak evidence documents.
    def _yield_text_leaves(value: Any, key_path: str) -> list[tuple[str, str]]:
        found: list[tuple[str, str]] = []
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                child_path = f"{key_path}.{sub_key}" if key_path else str(sub_key)
                found.extend(_yield_text_leaves(sub_value, child_path))
            return found
        if isinstance(value, list):
            for idx, item in enumerate(value):
                child_path = f"{key_path}[{idx}]"
                found.extend(_yield_text_leaves(item, child_path))
            return found
        if isinstance(value, str):
            text = value.strip()
            if len(text) >= 80:
                found.append((key_path or "metadata", text))
        return found

    for key, value in example.metadata.items():
        for key_path, text in _yield_text_leaves(value, key):
            rows.append(
                {
                    "id": stable_hash(example.example_id, key_path, text[:80]),
                    "title": f"{example.dataset_name} metadata {key_path}",
                    "text": text,
                    "source_type": "metadata",
                    "example_id": example.example_id,
                    "dataset": example.dataset_name,
                }
            )

    dedup: dict[str, dict[str, Any]] = {}
    for row in rows:
        dedup_key = stable_hash(row["id"], row["text"])
        dedup[dedup_key] = row
    return list(dedup.values())


def _load_examples_with_split_fallback(
    system: ConferenceEduSystem,
    dataset_name: str,
    source: str | None,
    preferred_split: str,
    limit: int | None,
) -> tuple[list[BenchmarkExample], str]:
    candidates = [preferred_split]
    for fallback in ["test", "validation", "train"]:
        if fallback not in candidates:
            candidates.append(fallback)

    last_error: Exception | None = None
    for split in candidates:
        try:
            examples = system.load_examples(dataset_name, source=source, split=split, limit=limit)
            return examples, split
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    # Last-resort fallback for datasets with inconsistent schemas on HF that
    # break load_dataset casting (observed on EduBench).
    try:
        manual_examples = _manual_snapshot_load(
            system=system,
            dataset_name=dataset_name,
            source=source,
            preferred_split=preferred_split,
            limit=limit,
        )
        if manual_examples:
            return manual_examples, "snapshot"
    except Exception:
        pass

    assert last_error is not None
    raise last_error


def _iter_rows_from_file(file_path: Path, preferred_split: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if file_path.suffix.lower() == ".jsonl":
        for line in file_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
            else:
                rows.append({"input": str(payload)})
        return rows

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if preferred_split and isinstance(payload.get(preferred_split), list):
            candidate_rows = payload[preferred_split]
        elif isinstance(payload.get("data"), list):
            candidate_rows = payload["data"]
        else:
            candidate_rows = [payload]
    elif isinstance(payload, list):
        candidate_rows = payload
    else:
        candidate_rows = [{"input": str(payload)}]

    for item in candidate_rows:
        if isinstance(item, dict):
            rows.append(item)
        else:
            rows.append({"input": str(item)})
    return rows


def _manual_snapshot_load(
    *,
    system: ConferenceEduSystem,
    dataset_name: str,
    source: str | None,
    preferred_split: str,
    limit: int | None,
) -> list[BenchmarkExample]:
    spec = system.dataset_registry.specs[dataset_name]
    adapter = system.dataset_registry.adapter_for(dataset_name)
    transform = getattr(adapter, "transform", generic_text_transform)
    dataset_id = source or spec.default_source
    if not dataset_id:
        return []

    snapshot_dir = snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        allow_patterns=["*.json", "*.jsonl"],
    )
    root = Path(snapshot_dir)
    files = sorted(
        [path for path in root.rglob("*") if path.suffix.lower() in {".json", ".jsonl"}],
        key=lambda p: (len(p.parts), str(p)),
    )

    examples: list[BenchmarkExample] = []
    lowered_split = preferred_split.lower()

    for file_path in files:
        rows = _iter_rows_from_file(file_path, preferred_split=preferred_split)
        for row in rows:
            split_value = str(row.get("split", row.get("Split", ""))).strip().lower()
            if split_value and lowered_split and split_value != lowered_split:
                continue

            # Common shape in benchmark exports: {"sample": {...}} or {"item": {...}}
            if len(row) == 1:
                only_value = next(iter(row.values()))
                if isinstance(only_value, dict):
                    row = only_value

            example = transform(row, spec)
            if not example.question.strip():
                continue
            examples.append(example)
            if limit is not None and len(examples) >= limit:
                return examples

    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize benchmark datasets into local JSONL files.")
    parser.add_argument("--config", default="configs/system.example.yaml")
    parser.add_argument("--datasets", nargs="+", default=["EduBench", "TutorEval"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Optional source override in the form DatasetName=source_identifier_or_path",
    )
    args = parser.parse_args()

    source_overrides: dict[str, str] = {}
    for item in args.source:
        if "=" not in item:
            raise ValueError(f"Invalid --source override: {item!r}; expected DatasetName=source")
        name, source = item.split("=", 1)
        source_overrides[name.strip()] = source.strip()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    system = ConferenceEduSystem(args.config)
    all_stats: dict[str, Any] = {}

    for dataset_name in args.datasets:
        source = source_overrides.get(dataset_name)
        examples, used_split = _load_examples_with_split_fallback(system, dataset_name, source, args.split, args.limit)

        dataset_dir = out_root / _slug(dataset_name)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        examples_path = dataset_dir / f"{used_split}.jsonl"
        corpus_path = dataset_dir / "corpus.jsonl"

        corpus_rows: list[dict[str, Any]] = []
        with examples_path.open("w", encoding="utf-8") as ex_fh:
            for example in examples:
                ex_fh.write(json.dumps(_example_to_row(example), ensure_ascii=False) + "\n")
                corpus_rows.extend(_extract_corpus_rows(example))

        dedup: dict[str, dict[str, Any]] = {}
        for row in corpus_rows:
            key = stable_hash(row.get("id"), row.get("text"))
            dedup[key] = row
        corpus_rows = list(dedup.values())

        with corpus_path.open("w", encoding="utf-8") as corpus_fh:
            for row in corpus_rows:
                corpus_fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        stats = {
            "dataset": dataset_name,
            "split": used_split,
            "source": source,
            "examples": len(examples),
            "corpus_docs": len(corpus_rows),
            "examples_path": str(examples_path),
            "corpus_path": str(corpus_path),
        }
        all_stats[dataset_name] = stats
        print(json.dumps(stats, ensure_ascii=False))

    summary_path = out_root / "dataset_prep_summary.json"
    summary_path.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()

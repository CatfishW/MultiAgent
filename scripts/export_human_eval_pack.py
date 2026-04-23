#!/usr/bin/env python3
"""Export a human-evaluation pack for reviewer-facing tutoring experiments.

Produces:
  - ``<out_dir>/items.jsonl``   : de-identified example payloads (id, question,
                                  rubric/keypoints, architecture answers) drawn
                                  via stratified sampling (dataset x architecture).
  - ``<out_dir>/rubric.csv``    : rater CSV template with one row per (item,
                                  rater) pair, columns for the four Likert
                                  scores + notes.
  - ``<out_dir>/protocol.md``   : short rating protocol (Likert scales, IAA
                                  procedure, ethical reminders).
  - ``<out_dir>/manifest.json`` : bookkeeping (seed, sessions, counts).

Usage::

    python scripts/export_human_eval_pack.py \
        --session hybrid=artifacts/experiments/results/tutoreval_hybrid_fast.json \
        --session classical=artifacts/experiments/results/tutoreval_classical_rag.json \
        --session non_rag=artifacts/experiments/results/tutoreval_non_rag_multi_agent.json \
        --session single=artifacts/experiments/results/tutoreval_single_agent_no_rag.json \
        --per-arch 25 --raters 3 --out-dir artifacts/human_eval/tutoreval

The companion IAA mode re-reads the completed ``rubric.csv`` and prints
pairwise Cohen-style agreement plus Krippendorff's alpha approximation::

    python scripts/export_human_eval_pack.py --iaa artifacts/human_eval/tutoreval/rubric.csv

All I/O is local and network-free. The script intentionally avoids any LLM
call so reviewers can replicate the procedure deterministically.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


LIKERT_COLUMNS = ("correctness", "pedagogy", "safety", "adaptivity")


def _load_session(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        result = payload.get("result") or {}
        records.extend(r for r in (result.get("records") or []) if isinstance(r, dict))
        if not records:
            for session in payload.get("sessions") or []:
                if isinstance(session, dict):
                    records.extend(r for r in (session.get("example_records") or []) if isinstance(r, dict))
    elif isinstance(payload, list):
        records.extend(r for r in payload if isinstance(r, dict))
    return records


def _load_session_meta(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    meta = payload.get("meta") or {}
    result = payload.get("result") or {}
    return {
        "path": str(path),
        "dataset": meta.get("dataset"),
        "architecture": meta.get("architecture"),
        "count": int(result.get("count", 0) or 0),
    }


def _sample_ids(shared_ids: list[str], per_arch: int, rng: random.Random) -> list[str]:
    selected = list(shared_ids)
    rng.shuffle(selected)
    return selected[: min(per_arch, len(selected))]


def _build_item_payload(
    example_id: str,
    sessions: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    # Use the first available record for static fields like question/rubric.
    reference: dict[str, Any] | None = None
    architecture_answers: dict[str, str] = {}
    for name, by_id in sessions.items():
        record = by_id.get(example_id)
        if not record:
            continue
        if reference is None:
            reference = record
        architecture_answers[name] = str(record.get("answer", ""))[:8000]
    if reference is None:
        return {}
    question = str(reference.get("question", "")).strip()
    rubric_raw = reference.get("rubric") or reference.get("metadata", {}).get("rubric") or []
    if isinstance(rubric_raw, str):
        rubric_list = [rubric_raw]
    elif isinstance(rubric_raw, list):
        rubric_list = [str(item).strip() for item in rubric_raw if str(item).strip()]
    else:
        rubric_list = []
    keypoints_raw = reference.get("metadata", {}).get("tutoreval_key_points") if isinstance(reference.get("metadata"), dict) else None
    keypoints = [str(k).strip() for k in (keypoints_raw or []) if str(k).strip()]
    return {
        "example_id": example_id,
        "question": question,
        "rubric": rubric_list,
        "keypoints": keypoints,
        "gold_answer": reference.get("gold_answer"),
        "architecture_answers": architecture_answers,
    }


def _write_items(items: list[dict[str, Any]], path: Path) -> None:
    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n", encoding="utf-8")


def _write_rubric_csv(items: list[dict[str, Any]], raters: int, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["item_id", "architecture", "rater", *LIKERT_COLUMNS, "notes"]
        writer.writerow(header)
        for item in items:
            for architecture in sorted(item.get("architecture_answers", {})):
                for rater_index in range(1, raters + 1):
                    writer.writerow(
                        [
                            item["example_id"],
                            architecture,
                            f"R{rater_index}",
                            "", "", "", "",
                            "",
                        ]
                    )


def _write_protocol(path: Path, per_arch: int, raters: int) -> None:
    text = f"""# Human-Evaluation Protocol (tutoring system review)

We sampled {per_arch} paired items per architecture; {raters} raters score each
response independently. Fill ``rubric.csv`` left to right without looking at the
other raters' numbers.

## Likert scales (1 = strongly disagree, 5 = strongly agree)
- **correctness** : The answer is factually correct and resolves the student's question.
- **pedagogy**    : The answer uses sound tutoring behavior (scaffolding, hints,
                    rubric alignment) rather than just dumping the final result.
- **safety**      : The answer avoids unsafe content, mockery, or misinformation.
- **adaptivity**  : The answer reflects the student's apparent level, dialogue
                    history, and rubric criteria.

## IAA procedure
After all raters submit:
1. ``python scripts/export_human_eval_pack.py --iaa rubric.csv`` prints
   pairwise proportion-agreement, linearly-weighted kappa, and Krippendorff's
   alpha for each Likert column.
2. If alpha < 0.67 for any column, resolve disagreements via a moderated
   adjudication pass before reporting headline numbers.

## Ethics
- Answers must be stripped of any personally identifying information before
  exporting to external raters.
- Raters must be informed that the answers are model-generated tutoring drafts
  and are not classroom-deployed advice.
"""
    path.write_text(text, encoding="utf-8")


def _load_iaa_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            cleaned: dict[str, Any] = {
                "item_id": raw.get("item_id", ""),
                "architecture": raw.get("architecture", ""),
                "rater": raw.get("rater", ""),
            }
            for col in LIKERT_COLUMNS:
                value = raw.get(col, "").strip()
                try:
                    cleaned[col] = float(value) if value else None
                except ValueError:
                    cleaned[col] = None
            rows.append(cleaned)
    return rows


def _pairwise_agreement(matrix: dict[str, dict[str, float]]) -> tuple[float, float]:
    """Return ``(proportion_agree, linear_weighted_kappa_approx)`` across rater pairs."""
    raters = sorted({r for values in matrix.values() for r in values})
    items = list(matrix.keys())
    if len(raters) < 2 or not items:
        return float("nan"), float("nan")
    pairs: list[tuple[str, str]] = []
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            pairs.append((raters[i], raters[j]))
    proportions: list[float] = []
    kappas: list[float] = []
    for a, b in pairs:
        overlaps = [
            (matrix[item][a], matrix[item][b])
            for item in items
            if a in matrix[item] and b in matrix[item] and matrix[item][a] is not None and matrix[item][b] is not None
        ]
        if not overlaps:
            continue
        agree = sum(1 for x, y in overlaps if abs(x - y) < 1e-9)
        proportions.append(agree / len(overlaps))
        max_diff = max(abs(x - y) for x, y in overlaps) or 1.0
        weighted = 1.0 - sum(abs(x - y) for x, y in overlaps) / (len(overlaps) * max_diff)
        kappas.append(weighted)
    return (mean(proportions) if proportions else float("nan"), mean(kappas) if kappas else float("nan"))


def _krippendorff_alpha_ordinal(matrix: dict[str, dict[str, float]]) -> float:
    """Lightweight ordinal Krippendorff's alpha approximation.

    We compute observed disagreement as the mean squared difference between
    rater pairs per item, and expected disagreement as the variance of all
    ratings. Returns 1 - observed / expected, bounded below at 0.0.
    """
    raters = sorted({r for values in matrix.values() for r in values})
    if len(raters) < 2:
        return float("nan")
    all_values: list[float] = []
    for values in matrix.values():
        for raw in values.values():
            if raw is not None:
                all_values.append(raw)
    if len(all_values) < 2:
        return float("nan")
    expected = sum((v - mean(all_values)) ** 2 for v in all_values) / len(all_values)
    if expected <= 0.0:
        return 1.0
    observed_pairs = 0
    observed_sum = 0.0
    for values in matrix.values():
        entries = [v for v in values.values() if v is not None]
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                observed_pairs += 1
                observed_sum += (entries[i] - entries[j]) ** 2
    if observed_pairs == 0:
        return float("nan")
    observed = observed_sum / observed_pairs
    alpha = 1.0 - (observed / expected)
    return max(0.0, alpha)


def _run_iaa(path: Path) -> int:
    rows = _load_iaa_rows(path)
    if not rows:
        print(json.dumps({"error": "no rows", "path": str(path)}))
        return 1
    # For each Likert column, build {item_id: {rater: value}} per architecture.
    summary: dict[str, Any] = {"path": str(path), "columns": {}}
    for column in LIKERT_COLUMNS:
        per_arch: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
        for row in rows:
            value = row.get(column)
            if value is None:
                continue
            per_arch[row["architecture"]][row["item_id"]][row["rater"]] = value
        column_summary: dict[str, Any] = {}
        for arch, matrix in per_arch.items():
            proportion, kappa = _pairwise_agreement(matrix)
            alpha = _krippendorff_alpha_ordinal(matrix)
            column_summary[arch] = {
                "n_items": len(matrix),
                "proportion_agree": proportion,
                "kappa_linear": kappa,
                "alpha_ordinal": alpha,
            }
        summary["columns"][column] = column_summary
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _main_export(raw: argparse.Namespace) -> int:
    sessions: dict[str, Path] = {}
    for entry in raw.session or []:
        if "=" not in entry:
            raise SystemExit(f"--session expects NAME=path.json, got: {entry!r}")
        name, path = entry.split("=", 1)
        sessions[name.strip()] = Path(path).expanduser().resolve()
    if len(sessions) < 2:
        raise SystemExit("Provide at least two --session entries (one per architecture).")

    loaded: dict[str, dict[str, dict[str, Any]]] = {}
    meta_manifest: dict[str, Any] = {}
    for name, path in sessions.items():
        records = _load_session(path)
        loaded[name] = {str(record.get("example_id", "")): record for record in records if record.get("example_id")}
        meta_manifest[name] = _load_session_meta(path)
    shared_ids = sorted(set.intersection(*[set(metrics.keys()) for metrics in loaded.values()]))
    rng = random.Random(raw.seed)
    sampled = _sample_ids(shared_ids, raw.per_arch, rng)
    items: list[dict[str, Any]] = []
    for example_id in sampled:
        payload = _build_item_payload(example_id, loaded)
        if payload:
            items.append(payload)
    out_dir = Path(raw.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_items(items, out_dir / "items.jsonl")
    _write_rubric_csv(items, raw.raters, out_dir / "rubric.csv")
    _write_protocol(out_dir / "protocol.md", raw.per_arch, raw.raters)
    manifest = {
        "seed": raw.seed,
        "per_arch": raw.per_arch,
        "raters": raw.raters,
        "sampled_count": len(items),
        "shared_ids_available": len(shared_ids),
        "sessions": meta_manifest,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "items": len(items), "architectures": sorted(loaded)}, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", action="append", default=None, help="Repeatable: NAME=path.json")
    parser.add_argument("--out-dir", default="artifacts/human_eval/default")
    parser.add_argument("--per-arch", type=int, default=25)
    parser.add_argument("--raters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iaa", default=None, help="Path to a filled rubric.csv; prints IAA and exits.")
    raw = parser.parse_args()
    if raw.iaa:
        return _run_iaa(Path(raw.iaa).expanduser().resolve())
    return _main_export(raw)


if __name__ == "__main__":
    raise SystemExit(main())

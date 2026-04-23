#!/usr/bin/env python3
"""Paired bootstrap confidence intervals and permutation tests.

Reads one session result JSON per architecture (written by ``run_eval_session``)
and produces paired statistics over the intersection of example IDs. Reviewers
can use the output TSV/JSONL to promote headline TutorEval/EduBench deltas from
point estimates to statistically grounded claims.

Inputs:
  - ``--session NAME=path.json`` arguments; NAME is a short label used in output
    rows (typically the architecture family, e.g. ``hybrid``).
  - Each session JSON must have ``result.records[].example_id`` and per-record
    ``metrics`` dict. Both full ``run_eval_session`` outputs and the dashboard
    ``session_summary.json`` slice are supported.

Outputs under ``--out-dir`` (default ``artifacts/stats/<label>/``):
  - ``pairs.jsonl``              : per-pair per-metric detail (mean delta, CI,
                                   p-value, n_paired).
  - ``pairs.tsv``                : wide summary for quick inspection.
  - ``inputs.json``              : bookkeeping about which sessions were parsed.

Statistical details:
  - ``compute_paired_stats`` performs a percentile-bootstrap on per-example
    differences at ``--confidence`` (default 0.95) with ``--bootstrap`` resamples
    (default 2000). Permutation test uses ``--permutations`` random sign flips
    (default 10000) to get a two-sided p-value for ``mean delta == 0``.
  - Missing metric values (NaN or absent) are dropped before pairing.

This script is network-free and does not re-evaluate anything; it only reads
existing result JSONs.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates: list[dict[str, Any]] = []
    # run_eval_session layout: result.records
    result = payload.get("result") if isinstance(payload, dict) else None
    if isinstance(result, dict):
        records = result.get("records")
        if isinstance(records, list):
            candidates.extend(r for r in records if isinstance(r, dict))
    # session_summary.json layout: sessions[].example_records
    if not candidates and isinstance(payload, dict):
        sessions = payload.get("sessions")
        if isinstance(sessions, list):
            for session in sessions:
                if not isinstance(session, dict):
                    continue
                example_records = session.get("example_records") or []
                if not isinstance(example_records, list):
                    continue
                candidates.extend(r for r in example_records if isinstance(r, dict))
    # direct JSONL fallback (list of dicts)
    if not candidates and isinstance(payload, list):
        candidates.extend(r for r in payload if isinstance(r, dict))
    return candidates


def _records_by_id(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for record in records:
        example_id = str(record.get("example_id", "")).strip()
        if not example_id:
            continue
        metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
        cleaned: dict[str, float] = {}
        for key, value in metrics.items():
            number = _safe_float(value)
            if number is None:
                continue
            cleaned[str(key)] = number
        out[example_id] = cleaned
    return out


def _bootstrap_ci(deltas: list[float], n_resamples: int, confidence: float, rng: random.Random) -> tuple[float, float]:
    if not deltas:
        return float("nan"), float("nan")
    n = len(deltas)
    means: list[float] = []
    for _ in range(max(1, n_resamples)):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    low_index = max(0, int(math.floor(alpha * len(means))))
    high_index = min(len(means) - 1, int(math.ceil((1.0 - alpha) * len(means))) - 1)
    return means[low_index], means[high_index]


def _permutation_pvalue(deltas: list[float], n_resamples: int, rng: random.Random) -> float:
    if not deltas:
        return float("nan")
    observed = abs(sum(deltas) / len(deltas))
    if observed == 0.0:
        return 1.0
    hits = 0
    n = len(deltas)
    total = max(1, n_resamples)
    for _ in range(total):
        flipped = [delta if rng.random() >= 0.5 else -delta for delta in deltas]
        if abs(sum(flipped) / n) >= observed:
            hits += 1
    return (hits + 1) / (total + 1)


def compute_paired_stats(
    a_metrics: dict[str, dict[str, float]],
    b_metrics: dict[str, dict[str, float]],
    *,
    metric_keys: Iterable[str] | None = None,
    bootstrap: int = 2000,
    permutations: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Return per-metric paired stats for ``(b - a)`` across the shared examples."""
    shared_ids = sorted(set(a_metrics) & set(b_metrics))
    if metric_keys is None:
        keys: set[str] = set()
        for record in list(a_metrics.values()) + list(b_metrics.values()):
            keys.update(record.keys())
        metric_keys = sorted(keys)
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for key in metric_keys:
        deltas: list[float] = []
        a_values: list[float] = []
        b_values: list[float] = []
        for eid in shared_ids:
            a_val = a_metrics.get(eid, {}).get(key)
            b_val = b_metrics.get(eid, {}).get(key)
            if a_val is None or b_val is None:
                continue
            deltas.append(b_val - a_val)
            a_values.append(a_val)
            b_values.append(b_val)
        if not deltas:
            rows.append(
                {
                    "metric": key,
                    "n_paired": 0,
                    "a_mean": float("nan"),
                    "b_mean": float("nan"),
                    "mean_delta": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "p_value": float("nan"),
                }
            )
            continue
        a_mean = mean(a_values)
        b_mean = mean(b_values)
        delta_mean = mean(deltas)
        ci_low, ci_high = _bootstrap_ci(deltas, bootstrap, confidence, rng)
        p_value = _permutation_pvalue(deltas, permutations, rng)
        rows.append(
            {
                "metric": key,
                "n_paired": len(deltas),
                "a_mean": a_mean,
                "b_mean": b_mean,
                "mean_delta": delta_mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p_value,
            }
        )
    return rows


def _write_outputs(out_dir: Path, rows: list[dict[str, Any]], inputs_meta: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pairs.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    headers = [
        "comparison",
        "metric",
        "n_paired",
        "a_mean",
        "b_mean",
        "mean_delta",
        "ci_low",
        "ci_high",
        "p_value",
    ]
    tsv_lines = ["\t".join(headers)]
    for row in rows:
        tsv_lines.append(
            "\t".join(
                [
                    str(row.get("comparison", "")),
                    str(row.get("metric", "")),
                    str(row.get("n_paired", 0)),
                    f"{row.get('a_mean', float('nan'))}",
                    f"{row.get('b_mean', float('nan'))}",
                    f"{row.get('mean_delta', float('nan'))}",
                    f"{row.get('ci_low', float('nan'))}",
                    f"{row.get('ci_high', float('nan'))}",
                    f"{row.get('p_value', float('nan'))}",
                ]
            )
        )
    (out_dir / "pairs.tsv").write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")
    (out_dir / "inputs.json").write_text(json.dumps(inputs_meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_session_specs(entries: list[str]) -> dict[str, Path]:
    sessions: dict[str, Path] = {}
    for entry in entries:
        if "=" not in entry:
            raise SystemExit(f"--session expects NAME=path.json, got: {entry!r}")
        name, path = entry.split("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit(f"--session entry missing a name before '=': {entry!r}")
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise SystemExit(f"Session file does not exist: {resolved}")
        sessions[name] = resolved
    return sessions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session",
        action="append",
        required=True,
        help="Repeatable: NAME=path/to/session.json. Provide at least two.",
    )
    parser.add_argument(
        "--compare",
        action="append",
        default=None,
        help="Repeatable: A,B -> bootstrap (B - A). Defaults to all pairs (i<j).",
    )
    parser.add_argument("--metric", action="append", default=None, help="Restrict to these metrics (repeatable).")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", default=None, help="Subdirectory name under --out-dir.")
    parser.add_argument("--out-dir", default="artifacts/stats")
    args = parser.parse_args()

    if len(args.session) < 2:
        raise SystemExit("Provide at least two --session entries for paired comparisons.")

    sessions = _parse_session_specs(args.session)
    session_records: dict[str, dict[str, dict[str, float]]] = {}
    inputs_meta: dict[str, Any] = {"sessions": {}, "compare": [], "metrics": args.metric or "<all>"}
    for name, path in sessions.items():
        records = _load_records(path)
        metrics_by_id = _records_by_id(records)
        session_records[name] = metrics_by_id
        inputs_meta["sessions"][name] = {"path": str(path), "n_examples": len(metrics_by_id)}

    pairs: list[tuple[str, str]] = []
    if args.compare:
        for entry in args.compare:
            if "," not in entry:
                raise SystemExit(f"--compare expects 'A,B', got: {entry!r}")
            a, b = (piece.strip() for piece in entry.split(",", 1))
            if a not in session_records or b not in session_records:
                raise SystemExit(f"Unknown session name in --compare {entry!r}: names are {sorted(session_records)}")
            pairs.append((a, b))
    else:
        names = list(sessions)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pairs.append((names[i], names[j]))

    inputs_meta["compare"] = [{"a": a, "b": b} for a, b in pairs]
    all_rows: list[dict[str, Any]] = []
    for a, b in pairs:
        rows = compute_paired_stats(
            session_records[a],
            session_records[b],
            metric_keys=args.metric,
            bootstrap=args.bootstrap,
            permutations=args.permutations,
            confidence=args.confidence,
            seed=args.seed,
        )
        for row in rows:
            row["comparison"] = f"{b} - {a}"
        all_rows.extend(rows)

    label = args.label or "paired"
    out_dir = Path(args.out_dir) / label
    _write_outputs(out_dir, all_rows, inputs_meta)
    print(json.dumps({"out_dir": str(out_dir), "rows": len(all_rows), "pairs": len(pairs)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

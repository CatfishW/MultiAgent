#!/usr/bin/env python3
"""Carve a stratified EduBench subset for fast re-runs.

Strategy: group by the sorted, joined set of ``metadata.information`` keys,
which in public EduBench exports is the closest practical proxy for a
scenario/task label (Subject/Level/Question vs Subject/Education Level/Anxiety
Level/Dialogue with Student, etc.). Allocate per-bucket quota proportional to
bucket frequency with a floor of 5 and a cap of 75. Deterministic seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _bucket_key(example: dict[str, Any]) -> str:
    meta = example.get("metadata") or {}
    info = meta.get("information") if isinstance(meta, dict) else None
    if isinstance(info, dict) and info:
        return "|".join(sorted(str(k) for k in info.keys()))
    return "__no_info__"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/processed/edubench/test.jsonl")
    ap.add_argument("--out", default="data/processed/edubench/test_subset500.jsonl")
    ap.add_argument("--manifest", default="data/processed/edubench/subset500_manifest.json")
    ap.add_argument("--target-size", type=int, default=500)
    ap.add_argument("--min-per-bucket", type=int, default=5)
    ap.add_argument("--max-per-bucket", type=int, default=75)
    ap.add_argument("--seed", type=int, default=20260420)
    args = ap.parse_args()

    src = Path(args.source)
    if not src.exists():
        print(f"source not found: {src}")
        return 1

    rng = random.Random(args.seed)

    buckets: dict[str, list[int]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    with src.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            rows.append(row)
            buckets[_bucket_key(row)].append(idx)

    total = len(rows)
    print(f"loaded {total} rows across {len(buckets)} buckets")

    bucket_order = sorted(buckets.items(), key=lambda kv: -len(kv[1]))
    # proportional allocation
    raw_allocs: dict[str, int] = {}
    for key, idxs in bucket_order:
        frac = len(idxs) / total
        raw = max(args.min_per_bucket, int(math.ceil(args.target_size * frac)))
        raw = min(raw, args.max_per_bucket, len(idxs))
        raw_allocs[key] = raw

    # rebalance down to target size by trimming largest allocations first
    overflow = sum(raw_allocs.values()) - args.target_size
    while overflow > 0:
        # trim the bucket with the largest current allocation above its floor
        candidate = max(
            (k for k, v in raw_allocs.items() if v > args.min_per_bucket),
            key=lambda k: raw_allocs[k],
            default=None,
        )
        if candidate is None:
            break
        raw_allocs[candidate] -= 1
        overflow -= 1

    sampled_idxs: list[int] = []
    bucket_counts: dict[str, int] = {}
    for key, alloc in raw_allocs.items():
        idxs = list(buckets[key])
        rng.shuffle(idxs)
        chosen = idxs[:alloc]
        sampled_idxs.extend(chosen)
        bucket_counts[key] = len(chosen)

    sampled_idxs.sort()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i in sampled_idxs:
            f.write(json.dumps(rows[i], ensure_ascii=False))
            f.write("\n")

    sampled_ids = [str(rows[i].get("id", "")) for i in sampled_idxs]
    manifest = {
        "source": str(src),
        "source_md5": _md5(src),
        "source_rows": total,
        "target_size": args.target_size,
        "actual_size": len(sampled_idxs),
        "seed": args.seed,
        "min_per_bucket": args.min_per_bucket,
        "max_per_bucket": args.max_per_bucket,
        "bucket_count": len(buckets),
        "bucket_counts": bucket_counts,
        "bucket_totals": {k: len(v) for k, v in buckets.items()},
        "sampled_ids": sampled_ids,
    }
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {out_path} with {len(sampled_idxs)} rows")
    print(f"manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

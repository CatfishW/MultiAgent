from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

from eduagentic.app import ConferenceEduSystem


def _serialize(value):
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Unsupported type: {type(value)!r}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run a benchmark dataset through the educational multi-agent framework.")
    parser.add_argument("dataset_name")
    parser.add_argument("--config", default=None)
    parser.add_argument("--source", default=None, help="Optional dataset source override (HF id or local file depending on adapter).")
    parser.add_argument("--split", default="test")
    parser.add_argument("--architecture", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="artifacts/benchmark_results.json")
    args = parser.parse_args()

    system = ConferenceEduSystem(args.config)
    results = await system.evaluate_dataset(
        args.dataset_name,
        source=args.source,
        split=args.split,
        architecture=args.architecture,
        limit=args.limit,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=_serialize), encoding="utf-8")
    print(f"Wrote results to {out_path.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())

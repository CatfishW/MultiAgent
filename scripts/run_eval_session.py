from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from eduagentic.app import ConferenceEduSystem
from eduagentic.retrieval.index import HybridIndex


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    return value


def _log(message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def _dataset_profile(examples: list[Any]) -> dict[str, Any]:
    total = len(examples)
    profile_counts: Counter[str] = Counter()
    with_gold = 0
    with_rubric = 0
    with_images = 0

    for example in examples:
        if getattr(example, "gold_answer", None):
            with_gold += 1
        rubric = getattr(example, "rubric", None)
        if rubric:
            with_rubric += 1
        images = getattr(example, "images", None)
        if images:
            with_images += 1
        metadata = getattr(example, "metadata", {}) or {}
        eval_profile = str(metadata.get("evaluation_profile", "generic"))
        profile_counts[eval_profile] += 1

    return {
        "total_examples": total,
        "with_gold_answer": with_gold,
        "with_rubric": with_rubric,
        "with_images": with_images,
        "evaluation_profiles": dict(profile_counts),
    }


def _metric_digest(summary: dict[str, Any]) -> str:
    keys = [
        "token_f1",
        "exact_match",
        "rubric_coverage",
        "edu_json_compliance",
        "edu_score_alignment",
        "grounded_overlap",
        "latency_ms",
    ]
    parts: list[str] = []
    for key in keys:
        if key not in summary:
            continue
        value = summary.get(key)
        if isinstance(value, (int, float)):
            parts.append(f"{key}={value:.4f}")
    return ", ".join(parts)


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    system = ConferenceEduSystem(args.config)

    _log("Initializing models from configured endpoints...")
    await system.initialize_models()
    _log(f"Selected text model: {system._deps.text_model}")
    _log(f"Selected vision model: {system._deps.vision_model}")
    if system._deps.text_chat_extra:
        _log(f"Text chat extra payload: {json.dumps(system._deps.text_chat_extra, ensure_ascii=False)}")
    if system._deps.vision_chat_extra:
        _log(f"Vision chat extra payload: {json.dumps(system._deps.vision_chat_extra, ensure_ascii=False)}")

    index_path = args.index_path
    if args.index_path:
        _log(f"Loading retrieval index from: {args.index_path}")
        index = HybridIndex.load(args.index_path)
        system._deps.retriever = index
        if system._pipelines is not None:
            for pipeline in system._pipelines.values():
                pipeline.deps.retriever = index
    elif args.corpus:
        _log(f"Building retrieval index from corpus: {args.corpus}")
        index = system.index_documents(args.corpus)
        if args.index_out:
            Path(args.index_out).mkdir(parents=True, exist_ok=True)
            index_file = index.save(args.index_out)
            index_path = str(index_file)
            _log(f"Saved index file: {index_path}")

    examples = system.load_examples(args.dataset_name, source=args.source, split=args.split, limit=args.limit)
    profile = _dataset_profile(examples)
    _log(
        "Running evaluation: "
        f"dataset={args.dataset_name}, split={args.split}, architecture={args.architecture}, "
        f"limit={args.limit}, examples={len(examples)}"
    )
    _log(f"Dataset supervision profile: {json.dumps(profile, ensure_ascii=False)}")

    results = await system.evaluator.evaluate(system, examples, architecture=args.architecture)
    duration_s = round(time.time() - started, 3)
    ended_at = time.strftime("%Y-%m-%d %H:%M:%S")

    payload = {
        "meta": {
            "dataset": args.dataset_name,
            "source": args.source,
            "split": args.split,
            "architecture": args.architecture,
            "limit": args.limit,
            "config": args.config,
            "corpus": args.corpus,
            "index_path": index_path,
            "duration_s": duration_s,
            "started_at": started_at,
            "ended_at": ended_at,
            "text_model": system._deps.text_model,
            "vision_model": system._deps.vision_model,
            "text_chat_extra": system._deps.text_chat_extra,
            "vision_chat_extra": system._deps.vision_chat_extra,
            "dataset_profile": profile,
        },
        "result": results,
    }
    if isinstance(results, dict):
        summary = results.get("summary", {}) or {}
        _log(f"Metric digest: {_metric_digest(summary)}")
    return payload


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run a logged benchmark evaluation session.")
    parser.add_argument("dataset_name")
    parser.add_argument("--config", default="configs/system.example.yaml")
    parser.add_argument("--source", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--architecture", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--corpus", default=None, help="Optional corpus path used to build retrieval index.")
    parser.add_argument("--index-path", default=None, help="Optional path to a prebuilt HybridIndex pickle file.")
    parser.add_argument("--index-out", default=None, help="Optional directory to save the built retrieval index.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    payload = await _run(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_serialize), encoding="utf-8")

    summary = payload["result"]["summary"] if isinstance(payload.get("result"), dict) else {}
    _log(f"Finished session. Wrote: {out_path.resolve()}")
    _log(f"Summary: {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    asyncio.run(main())

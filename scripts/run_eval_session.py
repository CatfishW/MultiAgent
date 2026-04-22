from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import tempfile
import time
from collections.abc import Mapping
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import httpx

from eduagentic.app import ConferenceEduSystem
from eduagentic.core.contracts import EvaluationRecord
from eduagentic.evaluation.metrics import canonical_answer_text, compute_metrics, summarize
from eduagentic.retrieval.index import HybridIndex


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(k): _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _log(message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


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
        "edubench_12d_mean",
        "edubench_scenario_adaptation",
        "edubench_factual_reasoning_accuracy",
        "edubench_pedagogical_application",
        "tutoreval_keypoint_hit_rate",
        "tutoreval_correctness",
        "tutoreval_completeness",
        "tutoreval_relevance",
        "tutoreval_keypoint_recall",
        "edu_json_compliance",
        "edu_score_alignment",
        "token_f1",
        "exact_match",
        "rubric_coverage",
        "grounded_overlap",
        "latency_ms",
        "api_time_ms",
        "llm_call_count",
        "total_tokens",
        "complexity_units",
    ]
    parts: list[str] = []
    for key in keys:
        if key not in summary:
            continue
        value = summary.get(key)
        if isinstance(value, (int, float)):
            parts.append(f"{key}={value:.4f}")
    return ", ".join(parts)


def _progress_digest(progress: dict[str, Any]) -> str:
    completed = int(progress.get("completed", 0))
    total = int(progress.get("total", 0))
    succeeded = int(progress.get("succeeded", 0))
    failed = int(progress.get("failed", 0))
    pct = (100.0 * completed / total) if total else 0.0
    rolling = progress.get("rolling_summary") if isinstance(progress.get("rolling_summary"), dict) else {}
    digest = _metric_digest(rolling)
    latest = str(progress.get("latest_example_id", "-"))
    return (
        f"Progress {completed}/{total} ({pct:.1f}%) "
        f"success={succeeded} failed={failed} latest={latest} | {digest}"
    )


def _retry_limit_for_exception(exc: Exception, *, default_retries: int, max_5xx_retries: int) -> int:
    retries = max(1, int(default_retries))
    if isinstance(exc, httpx.HTTPStatusError):
        status = int(exc.response.status_code)
        if 500 <= status < 600:
            return max(1, min(retries, int(max_5xx_retries)))
    return retries


def _resume_compatible(meta: dict[str, Any], args: argparse.Namespace) -> bool:
    checks = {
        "dataset": args.dataset_name,
        "split": args.split,
        "architecture": args.architecture,
    }
    if args.source is not None:
        checks["source"] = args.source
    for key, expected in checks.items():
        existing = meta.get(key)
        if existing is None or expected is None:
            continue
        if str(existing) != str(expected):
            return False
    return True


def _load_resume_state(
    out_path: Path,
    args: argparse.Namespace,
    allowed_example_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], str | None]:
    if not args.resume or not out_path.exists():
        return [], {}, None

    try:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
    except Exception as exc:
        _log(f"Resume disabled for unreadable output file ({out_path}): {exc}")
        return [], {}, None

    if not isinstance(existing, dict):
        _log("Resume disabled because existing payload is not a JSON object")
        return [], {}, None

    meta = existing.get("meta") if isinstance(existing.get("meta"), dict) else {}
    if meta and not _resume_compatible(meta, args):
        _log("Resume disabled due to metadata mismatch between output file and current run arguments")
        return [], {}, None

    result = existing.get("result") if isinstance(existing.get("result"), dict) else {}
    raw_records = result.get("records") if isinstance(result.get("records"), list) else []
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in raw_records:
        if not isinstance(item, dict):
            continue
        example_id = str(item.get("example_id", "")).strip()
        if not example_id or example_id in seen_ids or example_id not in allowed_example_ids:
            continue
        metrics = item.get("metrics")
        normalized_item = dict(item)
        normalized_item["metrics"] = metrics if isinstance(metrics, dict) else {}
        records.append(normalized_item)
        seen_ids.add(example_id)

    failures: dict[str, dict[str, Any]] = {}
    raw_failures = result.get("failures") if isinstance(result.get("failures"), list) else []
    for item in raw_failures:
        if not isinstance(item, dict):
            continue
        example_id = str(item.get("example_id", "")).strip()
        if not example_id or example_id not in allowed_example_ids:
            continue
        failures[example_id] = dict(item)

    started_at = str(meta.get("started_at")) if meta.get("started_at") else None
    _log(f"Resumed from existing output: reused_records={len(records)} pending_failures={len(failures)}")
    return records, failures, started_at


def _serialize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(record) for record in records]


def _build_payload(
    *,
    args: argparse.Namespace,
    index_path: str | None,
    profile: dict[str, Any],
    records: list[dict[str, Any]],
    failures: dict[str, dict[str, Any]],
    total_examples: int,
    started_at: str,
    initial_started_at: str | None,
    ended_at: str,
    duration_s: float,
    system: ConferenceEduSystem,
    resumed_records: int,
) -> dict[str, Any]:
    metric_rows = [record.get("metrics", {}) for record in records if isinstance(record.get("metrics"), dict)]
    summary = summarize(metric_rows) if metric_rows else {}
    failure_rows = [failures[key] for key in sorted(failures.keys())]
    processed_count = len(records) + len(failure_rows)
    return {
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
            "initial_started_at": initial_started_at or started_at,
            "ended_at": ended_at,
            "text_model": system._deps.text_model,
            "vision_model": system._deps.vision_model,
            "text_chat_extra": system._deps.text_chat_extra,
            "vision_chat_extra": system._deps.vision_chat_extra,
            "dataset_profile": profile,
            "resume_enabled": bool(args.resume),
            "resumed_records": resumed_records,
        },
        "result": {
            "count": len(records),
            "success_count": len(records),
            "processed_count": processed_count,
            "summary": summary,
            "records": _serialize_records(records),
            "failures": failure_rows,
            "failed_count": len(failure_rows),
            "total_examples": total_examples,
            "pending_examples": max(0, total_examples - processed_count),
        },
    }


def _checkpoint_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _serialize(payload)
    encoded = json.dumps(normalized, ensure_ascii=False, indent=2)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_file.write(encoded)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
            tmp_path = Path(tmp_file.name)
        tmp_path.replace(path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _retry_sleep_s(base_s: float, max_s: float, attempt: int) -> float:
    delay = min(max_s, base_s * (2 ** max(0, attempt - 1)))
    jitter = random.uniform(0.0, min(1.0, delay * 0.2))
    return max(0.0, delay + jitter)


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    system = ConferenceEduSystem(args.config)

    _log("Initializing models from configured endpoints...")
    await system.initialize_models()
    _log(f"Selected text model: {system._deps.text_model}")
    _log(f"Selected vision model: {system._deps.vision_model}")
    if system._deps.text_client is not None:
        _log(
            "Text client policy: "
            f"timeout_s={getattr(system._deps.text_client, 'timeout_s', '-')}, "
            f"request_retries={getattr(system._deps.text_client, 'request_retries', '-')}, "
            f"retry_base_s={getattr(system._deps.text_client, 'retry_base_s', '-')}, "
            f"retry_max_s={getattr(system._deps.text_client, 'retry_max_s', '-')}"
        )
    if system._deps.vision_client is not None:
        _log(
            "Vision client policy: "
            f"timeout_s={getattr(system._deps.vision_client, 'timeout_s', '-')}, "
            f"request_retries={getattr(system._deps.vision_client, 'request_retries', '-')}, "
            f"retry_base_s={getattr(system._deps.vision_client, 'retry_base_s', '-')}, "
            f"retry_max_s={getattr(system._deps.vision_client, 'retry_max_s', '-')}"
        )
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

    out_path = Path(args.out)
    allowed_ids = {str(example.example_id) for example in examples}
    resumed_records, failure_map, resumed_started_at = _load_resume_state(out_path, args, allowed_ids)
    records = list(resumed_records)
    seen_ids = {str(record.get("example_id", "")) for record in records}
    pending_examples = [example for example in examples if str(example.example_id) not in seen_ids]
    total_examples = len(examples)

    _log(
        "Execution mode: "
        f"resume={args.resume}, total_examples={total_examples}, "
        f"reused={len(records)}, pending={len(pending_examples)}, "
        f"max_example_retries={args.max_example_retries}"
    )

    # Write an initial checkpoint immediately so stale output from previous runs
    # does not appear in dashboards before the first example completes.
    initial_payload = _build_payload(
        args=args,
        index_path=index_path,
        profile=profile,
        records=records,
        failures=failure_map,
        total_examples=total_examples,
        started_at=started_at,
        initial_started_at=resumed_started_at,
        ended_at="",
        duration_s=round(time.time() - started, 3),
        system=system,
        resumed_records=len(resumed_records),
    )
    _checkpoint_write(out_path, initial_payload)

    writes_since_checkpoint = 0
    for example in pending_examples:
        last_exc: Exception | None = None
        completed_record: dict[str, Any] | None = None
        retry_limit_for_example = int(args.max_example_retries)

        for attempt in range(1, args.max_example_retries + 1):
            try:
                response = await asyncio.wait_for(
                    system.run_example(example, architecture=args.architecture),
                    timeout=args.example_timeout,
                )
                normalized_answer = canonical_answer_text(response.answer)
                metrics = compute_metrics(example, response)
                # Success + supervision flags for dashboard denominator-correct averages.
                metrics["success"] = 1.0
                metrics["has_gold"] = 1.0 if example.gold_answer else 0.0
                metrics["has_rubric"] = 1.0 if example.rubric else 0.0
                ref_score = example.metadata.get("edubench_reference_score_mean")
                metrics["has_reference_score"] = 1.0 if isinstance(ref_score, (int, float)) else 0.0
                record = EvaluationRecord(
                    example_id=example.example_id,
                    dataset_name=example.dataset_name,
                    architecture=response.architecture.value,
                    metrics=metrics,
                    answer=normalized_answer,
                    gold_answer=example.gold_answer,
                    retrieved_doc_ids=[chunk.doc_id for chunk in response.retrieved_chunks],
                )
                serialized = _serialize(record)
                if isinstance(serialized, dict):
                    serialized["success"] = True
                    serialized["evaluation_profile"] = str(example.metadata.get("evaluation_profile", "generic"))
                completed_record = serialized
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_exc = exc
                retry_limit = _retry_limit_for_exception(
                    exc,
                    default_retries=args.max_example_retries,
                    max_5xx_retries=args.max_5xx_retries,
                )
                retry_limit_for_example = retry_limit
                if attempt >= retry_limit:
                    break
                delay_s = _retry_sleep_s(args.retry_backoff_base, args.retry_backoff_max, attempt)
                _log(
                    f"Retrying example_id={example.example_id} "
                    f"attempt={attempt}/{retry_limit} "
                    f"after {delay_s:.1f}s due to {type(exc).__name__}: {exc}"
                )
                await asyncio.sleep(delay_s)

        if completed_record is not None:
            records.append(completed_record)
            seen_ids.add(str(example.example_id))
            failure_map.pop(str(example.example_id), None)
            writes_since_checkpoint += 1
        else:
            error_message = str(last_exc) if last_exc is not None else "Unknown error"
            failure_map[str(example.example_id)] = {
                "example_id": str(example.example_id),
                "error_type": type(last_exc).__name__ if last_exc is not None else "RuntimeError",
                "error": error_message,
                "attempts": int(retry_limit_for_example),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            _log(
                f"Failed example_id={example.example_id} after {retry_limit_for_example} attempts: "
                f"{failure_map[str(example.example_id)]['error_type']}: {error_message}"
            )
            writes_since_checkpoint += 1

        processed_count = len(records) + len(failure_map)
        if args.progress_every > 0 and (
            processed_count == total_examples
            or (processed_count > 0 and processed_count % args.progress_every == 0)
        ):
            metric_rows = [record.get("metrics", {}) for record in records if isinstance(record.get("metrics"), dict)]
            rolling_summary = summarize(metric_rows) if metric_rows else {}
            _log(
                _progress_digest(
                    {
                        "completed": processed_count,
                        "total": total_examples,
                        "succeeded": len(records),
                        "failed": len(failure_map),
                        "latest_example_id": example.example_id,
                        "rolling_summary": rolling_summary,
                    }
                )
            )

        if args.checkpoint_every > 0 and writes_since_checkpoint >= args.checkpoint_every:
            checkpoint_payload = _build_payload(
                args=args,
                index_path=index_path,
                profile=profile,
                records=records,
                failures=failure_map,
                total_examples=total_examples,
                started_at=started_at,
                initial_started_at=resumed_started_at,
                ended_at="",
                duration_s=round(time.time() - started, 3),
                system=system,
                resumed_records=len(resumed_records),
            )
            _checkpoint_write(out_path, checkpoint_payload)
            writes_since_checkpoint = 0

    duration_s = round(time.time() - started, 3)
    ended_at = time.strftime("%Y-%m-%d %H:%M:%S")
    payload = _build_payload(
        args=args,
        index_path=index_path,
        profile=profile,
        records=records,
        failures=failure_map,
        total_examples=total_examples,
        started_at=started_at,
        initial_started_at=resumed_started_at,
        ended_at=ended_at,
        duration_s=duration_s,
        system=system,
        resumed_records=len(resumed_records),
    )
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
    parser.add_argument("--progress-every", type=int, default=25, help="Emit rolling metric logs every N evaluated examples.")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True, help="Resume from existing --out file when present.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Start fresh and ignore existing --out file.")
    parser.add_argument("--max-example-retries", type=int, default=6, help="Retries per example before marking it failed.")
    parser.add_argument("--max-5xx-retries", type=int, default=2, help="Retry ceiling for HTTP 5xx failures.")
    parser.add_argument("--retry-backoff-base", type=float, default=2.0, help="Base seconds for example retry backoff.")
    parser.add_argument("--retry-backoff-max", type=float, default=45.0, help="Maximum seconds between example retries.")
    parser.add_argument("--checkpoint-every", type=int, default=25, help="Write checkpoint output every N processed examples.")
    parser.add_argument("--allow-partial", dest="allow_partial", action="store_true", default=True, help="Exit successfully even if some examples remain failed.")
    parser.add_argument("--no-allow-partial", dest="allow_partial", action="store_false", help="Fail with exit code 2 if some examples remain incomplete.")
    parser.add_argument("--example-timeout", type=float, default=300.0, help="Per-example timeout in seconds (default: 300).")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    payload = await _run(args)
    out_path = Path(args.out)
    _checkpoint_write(out_path, payload)

    result = payload.get("result", {}) if isinstance(payload, dict) else {}
    summary = result.get("summary", {}) if isinstance(result, dict) else {}
    total_examples = int(result.get("total_examples", 0)) if isinstance(result, dict) else 0
    completed = int(result.get("count", 0)) if isinstance(result, dict) else 0
    failed_count = int(result.get("failed_count", 0)) if isinstance(result, dict) else 0
    processed_count = int(result.get("processed_count", completed + failed_count)) if isinstance(result, dict) else 0
    complete = total_examples > 0 and completed >= total_examples and failed_count == 0

    _log(f"Finished session. Wrote: {out_path.resolve()}")
    _log(f"Summary: {json.dumps(summary, ensure_ascii=False)}")

    if not complete:
        _log(
            f"Session incomplete: processed={processed_count}/{total_examples}, "
            f"succeeded={completed}, failed_examples={failed_count}, "
            f"allow_partial={args.allow_partial}"
        )
        if not args.allow_partial:
            raise SystemExit(2)


if __name__ == "__main__":
    asyncio.run(main())

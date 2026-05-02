from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


TIMESTAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
PROGRESS_RE = re.compile(r"Progress\s+(?P<completed>\d+)/(?P<total>\d+)\s+\((?P<pct>[^\)]+)\)")
PROGRESS_METRIC_RE = re.compile(r"(?P<key>[a-zA-Z0-9_]+)=(?P<value>-?\d+(?:\.\d+)?)")
WRAPPED_START_RE = re.compile(
    r"START\s+(?P<tag>\S+)\s+dataset=(?P<dataset>\S+)\s+out=(?P<out>\S+)"
)
WRAPPED_DONE_RE = re.compile(r"DONE\s+(?P<tag>\S+)")


AGENTIC_RAG_RUNS = [
    {
        "tag": "qwen4b_tutoreval",
        "dataset": "TutorEval",
        "architecture": "agentic_rag",
        "result_file": "artifacts/experiments_agentic_rag/qwen4b/results/tutoreval_agentic_rag_n400.json",
        "backbone": "qwen4b",
    },
    {
        "tag": "qwen4b_edubench",
        "dataset": "EduBench",
        "architecture": "agentic_rag",
        "result_file": "artifacts/experiments_agentic_rag/qwen4b/results/edubench_agentic_rag_n400.json",
        "backbone": "qwen4b",
    },
    {
        "tag": "qwen27b_tutoreval",
        "dataset": "TutorEval",
        "architecture": "agentic_rag",
        "result_file": "artifacts/experiments_agentic_rag/qwen27b/results/tutoreval_agentic_rag_n400.json",
        "backbone": "qwen27b",
    },
    {
        "tag": "qwen27b_edubench",
        "dataset": "EduBench",
        "architecture": "agentic_rag",
        "result_file": "artifacts/experiments_agentic_rag/qwen27b/results/edubench_agentic_rag_n400.json",
        "backbone": "qwen27b",
    },
]

TOOLCALL_CACHE_RUNS = [
    {
        "tag": "marlet_cache_4b_tutoreval",
        "dataset": "TutorEval",
        "result_file": "artifacts/experiments_toolcall_50_cache/results/tutoreval_hybrid_fast_4b_tool_cache.json",
        "backbone": "qwen4b",
    },
    {
        "tag": "marlet_cache_4b_edubench",
        "dataset": "EduBench",
        "result_file": "artifacts/experiments_toolcall_50_cache/results/edubench_hybrid_fast_4b_tool_cache.json",
        "backbone": "qwen4b",
    },
    {
        "tag": "marlet_cache_27b_tutoreval",
        "dataset": "TutorEval",
        "result_file": "artifacts/experiments_toolcall_50_cache/results/tutoreval_hybrid_fast_27b_tool_cache.json",
        "backbone": "qwen27b",
    },
    {
        "tag": "marlet_cache_27b_edubench",
        "dataset": "EduBench",
        "result_file": "artifacts/experiments_toolcall_50_cache/results/edubench_hybrid_fast_27b_tool_cache.json",
        "backbone": "qwen27b",
    },
]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    # Result files are checkpointed during active runs while the dashboard
    # refresh loop polls every second. Retry briefly to avoid transient
    # JSONDecodeError when a checkpoint file is observed mid-write.
    for attempt in range(3):
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            if attempt < 2:
                time.sleep(0.05 * (attempt + 1))
                continue
            return None
        except Exception:
            return None
        return loaded if isinstance(loaded, dict) else None
    return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _extract_json_suffix(line: str, marker: str) -> dict[str, Any] | None:
    if marker not in line:
        return None
    payload = line.split(marker, 1)[1].strip()
    try:
        parsed = json.loads(payload)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_pct_text(value: str | None) -> float:
    if not value:
        return 0.0
    cleaned = value.strip().replace("%", "")
    try:
        return float(cleaned)
    except Exception:
        return 0.0


def _parse_progress_metrics(line: str) -> dict[str, float]:
    if "|" not in line:
        return {}
    suffix = line.split("|", 1)[1]
    metrics: dict[str, float] = {}
    for match in PROGRESS_METRIC_RE.finditer(suffix):
        metrics[match.group("key")] = _safe_float(match.group("value"), 0.0)
    return metrics


def _parse_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = text.splitlines()

    timestamps = [match.group(1) for line in lines if (match := TIMESTAMP_RE.match(line))]
    started_at = timestamps[0] if timestamps else None
    ended_at = timestamps[-1] if timestamps else None

    finished = any("Finished session." in line for line in lines)
    wrapper_succeeded = any("succeeded on attempt" in line for line in lines)
    wrapper_exhausted = any("exhausted retries" in line for line in lines)
    error_lines = [line for line in lines if "Traceback" in line or "ERROR" in line or "Exception" in line]
    has_error = bool(error_lines)

    summary_line = next((line for line in reversed(lines) if "Summary:" in line), None)
    metric_digest = next((line.split("Metric digest:", 1)[1].strip() for line in reversed(lines) if "Metric digest:" in line), None)

    latest_progress = None
    for line in reversed(lines):
        match = PROGRESS_RE.search(line)
        if not match:
            continue
        pct_text = match.group("pct")
        latest_progress = {
            "completed": int(match.group("completed")),
            "total": int(match.group("total")),
            "pct": _parse_pct_text(pct_text),
            "pct_text": pct_text,
            "metrics": _parse_progress_metrics(line),
            "line": line,
        }
        break

    supervision_profile = None
    for line in reversed(lines):
        maybe = _extract_json_suffix(line, "Dataset supervision profile:")
        if maybe is not None:
            supervision_profile = maybe
            break

    timeline = [
        line
        for line in lines
        if any(
            marker in line
            for marker in [
                "Initializing models",
                "Selected text model",
                "Selected vision model",
                "Loading retrieval index",
                "Building retrieval index",
                "Running evaluation",
                "Dataset supervision profile",
                "Metric digest",
                "Finished session",
            ]
        )
    ]

    return {
        "started_at": started_at,
        "ended_at": ended_at,
        "finished": finished,
        "wrapper_succeeded": wrapper_succeeded,
        "wrapper_exhausted": wrapper_exhausted,
        "has_error": has_error,
        "error_lines": error_lines[-6:],
        "summary_line": summary_line,
        "metric_digest": metric_digest,
        "latest_progress": latest_progress,
        "supervision_profile": supervision_profile,
        "line_count": len(lines),
        "timeline": timeline[-12:],
        "tail": lines[-12:],
    }


def _session_name_to_key(log_name: str) -> str:
    base = log_name.replace("exp_", "").replace(".log", "")
    return base


def _split_dataset_arch(session_key: str) -> tuple[str, str]:
    if session_key.startswith("ablation_"):
        rest = session_key[len("ablation_"):]
        for prefix, dataset_name in (("edubench_", "EduBench"), ("tutoreval_", "TutorEval")):
            idx = rest.find(prefix)
            if idx >= 0:
                tag = rest[:idx].rstrip("_")
                arch = rest[idx + len(prefix):]
                return f"{dataset_name} (ablation: {tag})", f"{arch} [{tag}]"
        if "_" in rest:
            tag, _, remainder = rest.partition("_")
            return f"Ablation ({tag})", remainder
        return "Ablation", rest
    if session_key.startswith("edubench_"):
        return "EduBench", session_key[len("edubench_") :]
    if session_key.startswith("tutoreval_"):
        return "TutorEval", session_key[len("tutoreval_") :]
    if "_" in session_key:
        first, rest = session_key.split("_", 1)
        return first, rest
    return session_key, "unknown"


def _score(summary: dict[str, Any]) -> float:
    weights = {
        "edubench_12d_mean": 0.22,
        "edubench_scenario_adaptation": 0.08,
        "edubench_factual_reasoning_accuracy": 0.1,
        "edubench_pedagogical_application": 0.1,
        "tutoreval_keypoint_hit_rate": 0.2,
        "tutoreval_correctness": 0.1,
        "tutoreval_completeness": 0.08,
        "tutoreval_relevance": 0.07,
        "tutoreval_keypoint_recall": 0.06,
        "rubric_coverage": 0.05,
        "grounded_overlap": 0.05,
        "citation_coverage": 0.04,
        "edu_json_compliance": 0.04,
        "edu_score_alignment": 0.03,
        "token_f1": 0.04,
        "exact_match": 0.02,
    }

    weighted_sum = 0.0
    weight_total = 0.0
    for key, weight in weights.items():
        if key in summary:
            weighted_sum += _safe_float(summary.get(key), 0.0) * weight
            weight_total += weight

    latency = _safe_float(summary.get("latency_ms"), 0.0)
    if latency > 0:
        speed_bonus = 1.0 / (1.0 + latency / 4000.0)
        weighted_sum += speed_bonus * 0.05
        weight_total += 0.05

    api_time = _safe_float(summary.get("api_time_ms"), 0.0)
    if api_time > 0:
        api_efficiency = 1.0 / (1.0 + api_time / 3000.0)
        weighted_sum += api_efficiency * 0.03
        weight_total += 0.03

    complexity_units = _safe_float(summary.get("complexity_units"), 0.0)
    if complexity_units > 0:
        complexity_efficiency = 1.0 / (1.0 + complexity_units / 6000.0)
        weighted_sum += complexity_efficiency * 0.03
        weight_total += 0.03

    if weight_total == 0.0:
        return 0.0
    return weighted_sum / weight_total


def _metric_tiles(summary: dict[str, Any]) -> dict[str, float]:
    # Always include per-key denominator (``<key>_n``) when present so the dashboard
    # can show "mean (n=X)" transparently for NaN-skipped metrics.
    keys = [
        "edubench_12d_mean",
        "edubench_scenario_adaptation",
        "edubench_factual_reasoning_accuracy",
        "edubench_pedagogical_application",
        "edubench_iftc",
        "edubench_rtc",
        "edubench_crsc",
        "edubench_sei",
        "edubench_bfa",
        "edubench_dka",
        "edubench_rpr",
        "edubench_eicp",
        "edubench_csi",
        "edubench_mgp",
        "edubench_pas",
        "edubench_hots",
        "tutoreval_keypoint_hit_rate",
        "tutoreval_correctness",
        "tutoreval_completeness",
        "tutoreval_relevance",
        "token_f1",
        "exact_match",
        "rubric_coverage",
        "tutoreval_keypoint_recall",
        "grounded_overlap",
        "citation_coverage",
        "edu_json_compliance",
        "edu_score_alignment",
        "latency_ms",
        "api_time_ms",
        "model_cache_hits",
        "agent_count",
        "llm_call_count",
        "total_tokens",
        "complexity_units",
    ]
    tiles: dict[str, float] = {}
    for key in keys:
        if key in summary:
            tiles[key] = round(_safe_float(summary.get(key), 0.0), 4)
        n_key = f"{key}_n"
        if n_key in summary:
            tiles[n_key] = _safe_float(summary.get(n_key), 0.0)
    # include supervision flag means to expose coverage
    for key in ("success", "has_gold", "has_rubric", "has_reference_score"):
        if key in summary:
            tiles[key] = round(_safe_float(summary.get(key), 0.0), 4)
    return tiles


def _extract_thinking_budget(meta: dict[str, Any]) -> int | None:
    text_extra = meta.get("text_chat_extra")
    if isinstance(text_extra, dict):
        value = text_extra.get("extra_body", {}).get("thinking_budget")
        if isinstance(value, int):
            return value
        value = text_extra.get("qwen_reasoning_budget")
        if isinstance(value, int):
            return value
    vision_extra = meta.get("vision_chat_extra")
    if isinstance(vision_extra, dict):
        value = vision_extra.get("extra_body", {}).get("thinking_budget")
        if isinstance(value, int):
            return value
        value = vision_extra.get("qwen_reasoning_budget")
        if isinstance(value, int):
            return value
    return None


def _locate_result_file(results_dir: Path, ablation_results_dir: Path, session_key: str) -> Path:
    direct = results_dir / f"{session_key}.json"
    if direct.exists():
        return direct
    nested = sorted(results_dir.rglob(f"{session_key}.json"))
    if nested:
        return nested[0]
    if session_key.startswith("ablation_"):
        rest = session_key[len("ablation_"):]
        for prefix in ("edubench_", "tutoreval_"):
            idx = rest.find(prefix)
            if idx >= 0:
                tag = rest[:idx].rstrip("_")
                arch_part = rest[idx + len(prefix):]
                candidates = [
                    ablation_results_dir / tag / f"{prefix}{arch_part}.json",
                    ablation_results_dir / tag / f"{arch_part}.json",
                ]
                for candidate in candidates:
                    if candidate.exists():
                        return candidate
    return direct


def _normalize_progress(progress: dict[str, Any] | None, *, status: str, record_count: int) -> dict[str, Any] | None:
    if isinstance(progress, dict):
        completed = int(progress.get("completed") or 0)
        total = int(progress.get("total") or 0)
        pct = _safe_float(progress.get("pct"), 0.0)
        pct_text = str(progress.get("pct_text") or "")
    else:
        completed = 0
        total = 0
        pct = 0.0
        pct_text = ""

    if status == "finished" and record_count > 0:
        if total <= 0:
            total = record_count
        completed = max(completed, record_count)

    if total > 0 and pct <= 0.0:
        pct = (100.0 * completed / total) if total else 0.0

    if status == "finished" and total > 0 and completed >= total:
        pct = 100.0

    if total <= 0 and completed <= 0:
        return None

    return {
        "completed": completed,
        "total": total,
        "pct": round(pct, 2),
        "pct_text": pct_text if pct_text else f"{pct:.1f}%",
    }


def _extract_records_sample(result_payload: dict[str, Any] | None, max_n: int = 50) -> list[dict[str, Any]]:
    """Extract a limited sample of per-example records from a result payload."""
    if not result_payload or not isinstance(result_payload, dict):
        return []
    result = result_payload.get("result")
    if not isinstance(result, dict):
        return []
    records = result.get("records", [])
    if not isinstance(records, list):
        return []
    sample = records[:max_n]
    out: list[dict[str, Any]] = []
    for r in sample:
        if not isinstance(r, dict):
            continue
        metrics = r.get("metrics", {}) if isinstance(r.get("metrics"), dict) else {}
        out.append(
            {
                "example_id": r.get("example_id", ""),
                "question": r.get("question", "")[:200] if isinstance(r.get("question"), str) else "",
                "answer": (r.get("answer") or r.get("normalized_answer") or "")[:200] if isinstance((r.get("answer") or r.get("normalized_answer")), str) else "",
                "success": bool(r.get("success", False)),
                "metrics": {
                    "token_f1": metrics.get("token_f1"),
                    "exact_match": metrics.get("exact_match"),
                    "rubric_coverage": metrics.get("rubric_coverage"),
                },
            }
        )
    return out


def _extract_progress_history(log_info: dict[str, Any]) -> list[float] | None:
    """Extract progress percentage points from log timeline for sparklines."""
    timeline = log_info.get("timeline") or []
    if not isinstance(timeline, list) or len(timeline) < 3:
        return None
    # Parse "Progress X/Y (Z%)" lines
    pcts: list[float] = []
    for line in timeline:
        m = re.search(r"Progress\s+\d+/\d+\s+\((\d+(?:\.\d+)?)%\)", str(line))
        if m:
            pcts.append(float(m.group(1)))
    return pcts if len(pcts) >= 2 else None


def _progress_history_from_lines(lines: list[str]) -> list[float] | None:
    pcts: list[float] = []
    for line in lines:
        match = PROGRESS_RE.search(line)
        if match:
            pcts.append(_parse_pct_text(match.group("pct")))
    return pcts if len(pcts) >= 2 else None


def _combined_log_segments(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    segments: dict[str, dict[str, Any]] = {}
    current_tag: str | None = None
    for line in lines:
        if (match := WRAPPED_START_RE.search(line)):
            current_tag = match.group("tag")
            segments[current_tag] = {
                "dataset": match.group("dataset"),
                "result_file": match.group("out"),
                "started_at": TIMESTAMP_RE.match(line).group(1) if TIMESTAMP_RE.match(line) else None,
                "ended_at": None,
                "done": False,
                "lines": [line],
            }
            continue
        if (match := WRAPPED_DONE_RE.search(line)):
            tag = match.group("tag")
            segment = segments.setdefault(tag, {"lines": []})
            segment.setdefault("lines", []).append(line)
            segment["ended_at"] = TIMESTAMP_RE.match(line).group(1) if TIMESTAMP_RE.match(line) else None
            segment["done"] = True
            if current_tag == tag:
                current_tag = None
            continue
        if current_tag and current_tag in segments:
            segments[current_tag].setdefault("lines", []).append(line)
    return segments


def _agentic_rag_log_segments(root_dir: Path) -> tuple[dict[str, dict[str, Any]], float, dict[str, Path]]:
    log_dir = root_dir / "logs/experiments/agentic_rag"
    if not log_dir.exists():
        return {}, 0.0, {}

    segments: dict[str, dict[str, Any]] = {}
    source_files: dict[str, Path] = {}
    latest_mtime = 0.0
    for log_file in sorted(log_dir.glob("*.log")):
        latest_mtime = max(latest_mtime, log_file.stat().st_mtime)
        for tag, segment in _combined_log_segments(log_file).items():
            existing = segments.get(tag)
            existing_len = len(existing.get("lines", [])) if isinstance(existing, dict) else -1
            segment_len = len(segment.get("lines", []))
            # Prefer the log that actually contains the active/completed run for a tag.
            if existing is None or segment.get("done") or segment_len >= existing_len:
                segments[tag] = segment
                source_files[tag] = log_file
    return segments, latest_mtime, source_files


def _append_agentic_rag_sessions(
    *,
    root_dir: Path,
    sessions: list[dict[str, Any]],
    by_dataset: dict[str, list[dict[str, Any]]],
) -> None:
    segments, latest_log_mtime, source_files = _agentic_rag_log_segments(root_dir)
    if not segments:
        return
    active_tags = set()
    for idx, spec in enumerate(AGENTIC_RAG_RUNS):
        tag = spec["tag"]
        segment = segments.get(tag, {})
        lines = list(segment.get("lines") or [])
        log_file = source_files.get(tag, root_dir / "logs/experiments/agentic_rag/both_models.log")
        log_mtime = log_file.stat().st_mtime if log_file.exists() else latest_log_mtime
        result_file = root_dir / spec["result_file"]
        result_payload = _load_json(result_file)
        result = result_payload.get("result", {}) if isinstance(result_payload, dict) else {}
        meta = result_payload.get("meta", {}) if isinstance(result_payload, dict) else {}
        summary = result.get("summary", {}) or {} if isinstance(result, dict) else {}
        success_count = int(result.get("count", 0) or 0) if isinstance(result, dict) else 0
        failed_count = int(result.get("failed_count", 0) or 0) if isinstance(result, dict) else 0
        record_count = int(result.get("processed_count", success_count + failed_count) or 0) if isinstance(result, dict) else 0
        total_examples = int(result.get("total_examples", 400) or 400) if isinstance(result, dict) else 400

        latest_progress = None
        for line in reversed(lines):
            match = PROGRESS_RE.search(line)
            if match:
                latest_progress = {
                    "completed": int(match.group("completed")),
                    "total": int(match.group("total")),
                    "pct": _parse_pct_text(match.group("pct")),
                    "pct_text": match.group("pct"),
                    "metrics": _parse_progress_metrics(line),
                    "line": line,
                }
                break
        progress = _normalize_progress(latest_progress, status="running", record_count=record_count)
        if progress is None:
            pct = (100.0 * record_count / total_examples) if total_examples else 0.0
            progress = {
                "completed": record_count,
                "total": total_examples,
                "pct": round(pct, 2),
                "pct_text": f"{pct:.1f}%",
            }
        if latest_progress and isinstance(latest_progress.get("metrics"), dict):
            summary = {**summary, **latest_progress["metrics"]}
            record_count = max(record_count, int(latest_progress.get("completed") or 0))
            if progress and total_examples > 0 and record_count > int(progress.get("completed", 0) or 0):
                pct = 100.0 * record_count / total_examples
                progress = {
                    "completed": record_count,
                    "total": total_examples,
                    "pct": round(pct, 2),
                    "pct_text": f"{pct:.1f}%",
                }

        done = bool(segment.get("done"))
        complete_by_result = total_examples > 0 and success_count + failed_count >= total_examples
        previous_complete = all(
            (root_dir / prior["result_file"]).exists()
            and ((_load_json(root_dir / prior["result_file"]) or {}).get("result", {}).get("count", 0) or 0) >= 400
            for prior in AGENTIC_RAG_RUNS[:idx]
        )
        status = "finished" if done or complete_by_result else "running" if lines or previous_complete else "queued"
        if status == "running":
            active_tags.add(tag)
        progress_ratio = _safe_float(progress.get("pct"), 0.0) / 100.0 if progress else 0.0
        score = _score(summary) if summary else 0.0
        model_from_spec = "Qwen3.5-27B-FP8" if spec["backbone"] == "qwen27b" else "Qwen3.5-4B"

        session = {
            "session_key": f"{spec['backbone']}__{spec['dataset'].lower()}_agentic_rag_n400",
            "dataset": spec["dataset"],
            "architecture": "agentic_rag",
            "status": status,
            "started_at": segment.get("started_at") or meta.get("started_at"),
            "ended_at": meta.get("ended_at") or segment.get("ended_at"),
            "records": record_count,
            "success_records": success_count,
            "failed_records": failed_count,
            "summary": summary,
            "metric_tiles": _metric_tiles(summary),
            "status_reason": "complete" if status == "finished" else ("queued after earlier Agentic RAG runs" if status == "queued" else f"screen log: {log_file.name}"),
            "score": round(score, 4),
            "result_file": str(result_file),
            "log_file": str(log_file),
            "meta": meta,
            "models": {
                "text": meta.get("text_model") or model_from_spec,
                "vision": meta.get("vision_model") or model_from_spec,
            },
            "thinking_budget": _extract_thinking_budget(meta) or 512,
            "duration_s": round(_safe_float(meta.get("duration_s"), 0.0), 3) if meta.get("duration_s") is not None else None,
            "supervision_profile": meta.get("dataset_profile") if isinstance(meta.get("dataset_profile"), dict) else None,
            "result_fresh": result_file.exists() and (not lines or result_file.stat().st_mtime >= log_mtime - 1.0),
            "progress": progress,
            "progress_ratio": round(progress_ratio, 4),
            "metric_digest": None,
            "log_tail": lines[-12:],
            "log_timeline": [
                line
                for line in lines
                if any(marker in line for marker in ["START", "Selected text model", "Running evaluation", "Progress", "DONE"])
            ][-12:],
            "log_line_count": len(lines),
            "log_errors": [line for line in lines if "Traceback" in line or "ERROR" in line or "Exception" in line][-6:],
            "example_records": _extract_records_sample(result_payload),
            "progress_history": _progress_history_from_lines(lines),
            "backbone": spec["backbone"],
        }
        sessions.append(session)
        by_dataset[spec["dataset"]].append(session)


def _append_toolcall_cache_sessions(
    *,
    root_dir: Path,
    sessions: list[dict[str, Any]],
    by_dataset: dict[str, list[dict[str, Any]]],
) -> None:
    log_dir = root_dir / "logs/experiments/toolcall_50_cache"
    driver_log = log_dir / "driver.log"
    segments = _combined_log_segments(driver_log)
    latest_log_mtime = driver_log.stat().st_mtime if driver_log.exists() else 0.0
    for spec in TOOLCALL_CACHE_RUNS:
        tag = spec["tag"]
        segment = segments.get(tag, {})
        result_file = root_dir / spec["result_file"]
        result_payload = _load_json(result_file)
        result = result_payload.get("result", {}) if isinstance(result_payload, dict) else {}
        meta = result_payload.get("meta", {}) if isinstance(result_payload, dict) else {}
        summary = result.get("summary", {}) or {} if isinstance(result, dict) else {}
        success_count = int(result.get("count", 0) or 0) if isinstance(result, dict) else 0
        failed_count = int(result.get("failed_count", 0) or 0) if isinstance(result, dict) else 0
        record_count = int(result.get("processed_count", success_count + failed_count) or 0) if isinstance(result, dict) else 0
        total_examples = int(result.get("total_examples", 50) or 50) if isinstance(result, dict) else 50

        lines = list(segment.get("lines") or [])
        run_log = log_dir / f"{tag}.log"
        if run_log.exists():
            run_lines = run_log.read_text(encoding="utf-8", errors="ignore").splitlines()
            lines = [*lines, *run_lines]
            latest_log_mtime = max(latest_log_mtime, run_log.stat().st_mtime)
        latest_progress = None
        for line in reversed(lines):
            if match := PROGRESS_RE.search(line):
                latest_progress = {
                    "completed": int(match.group("completed")),
                    "total": int(match.group("total")),
                    "pct": _parse_pct_text(match.group("pct")),
                    "pct_text": match.group("pct"),
                    "metrics": _parse_progress_metrics(line),
                    "line": line,
                }
                break
        if latest_progress and isinstance(latest_progress.get("metrics"), dict):
            summary = {**summary, **latest_progress["metrics"]}
            record_count = max(record_count, int(latest_progress.get("completed") or 0))
        done = bool(segment.get("done"))
        complete_by_result = total_examples > 0 and success_count + failed_count >= total_examples
        status = "finished" if done or complete_by_result else "running" if lines or record_count else "queued"
        progress = _normalize_progress(latest_progress, status=status, record_count=record_count)
        if progress is None:
            pct = (100.0 * record_count / total_examples) if total_examples else 0.0
            progress = {"completed": record_count, "total": total_examples, "pct": round(pct, 2), "pct_text": f"{pct:.1f}%"}
        progress_ratio = _safe_float(progress.get("pct"), 0.0) / 100.0 if progress else 0.0
        model_from_spec = "Qwen3.5-27B-FP8" if spec["backbone"] == "qwen27b" else "Qwen3.5-4B"
        score = _score(summary) if summary else 0.0
        log_mtime = max(latest_log_mtime, run_log.stat().st_mtime if run_log.exists() else 0.0)
        session = {
            "session_key": f"{spec['backbone']}__{spec['dataset'].lower()}_marlet_tool_cache_50",
            "dataset": spec["dataset"],
            "architecture": "hybrid_fast_tool_cache",
            "status": status,
            "started_at": segment.get("started_at") or meta.get("started_at"),
            "ended_at": meta.get("ended_at") or segment.get("ended_at"),
            "records": record_count,
            "success_records": success_count,
            "failed_records": failed_count,
            "summary": summary,
            "metric_tiles": _metric_tiles(summary),
            "status_reason": "complete" if status == "finished" else "cache-aware tool-call MARLET run",
            "score": round(score, 4),
            "result_file": str(result_file),
            "log_file": str(run_log if run_log.exists() else driver_log),
            "meta": meta,
            "models": {
                "text": meta.get("text_model") or model_from_spec,
                "vision": meta.get("vision_model") or model_from_spec,
            },
            "thinking_budget": _extract_thinking_budget(meta) or 512,
            "duration_s": round(_safe_float(meta.get("duration_s"), 0.0), 3) if meta.get("duration_s") is not None else None,
            "supervision_profile": meta.get("dataset_profile") if isinstance(meta.get("dataset_profile"), dict) else None,
            "result_fresh": result_file.exists() and (not lines or result_file.stat().st_mtime >= log_mtime - 1.0),
            "progress": progress,
            "progress_ratio": round(progress_ratio, 4),
            "metric_digest": None,
            "log_tail": lines[-12:],
            "log_timeline": [
                line
                for line in lines
                if any(marker in line for marker in ["START", "Selected text model", "Running evaluation", "Progress", "DONE"])
            ][-12:],
            "log_line_count": len(lines),
            "log_errors": [line for line in lines if "Traceback" in line or "ERROR" in line or "Exception" in line][-6:],
            "example_records": _extract_records_sample(result_payload),
            "progress_history": _progress_history_from_lines(lines),
            "backbone": spec["backbone"],
        }
        sessions.append(session)
        by_dataset[spec["dataset"]].append(session)


def _architecture_leaderboard(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for session in sessions:
        if session.get("status") != "finished":
            continue
        grouped[str(session.get("architecture"))].append(_safe_float(session.get("score"), 0.0))

    rows = []
    for architecture, scores in grouped.items():
        if not scores:
            continue
        rows.append(
            {
                "architecture": architecture,
                "runs": len(scores),
                "avg_score": round(sum(scores) / len(scores), 4),
                "best_score": round(max(scores), 4),
            }
        )
    rows.sort(key=lambda item: item["avg_score"], reverse=True)
    return rows


def _scan_track(
    *,
    logs_dir: Path,
    results_dir: Path,
    ablation_results_dir: Path,
    backbone_tag: str,
    is_primary: bool,
    sessions: list[dict[str, Any]],
    by_dataset: dict[str, list[dict[str, Any]]],
    known_log_keys: set[str],
) -> None:
    if not logs_dir.exists():
        return
    for log_file in sorted(logs_dir.rglob("exp_*.log")):
        raw_key = _session_name_to_key(log_file.name)
        session_key = raw_key if is_primary else f"{backbone_tag}__{raw_key}"
        known_log_keys.add(session_key)
        dataset, architecture = _split_dataset_arch(raw_key)
        result_file = _locate_result_file(results_dir, ablation_results_dir, raw_key)
        result_exists = result_file.exists()
        result_mtime = result_file.stat().st_mtime if result_exists else 0.0
        log_mtime = log_file.stat().st_mtime if log_file.exists() else 0.0

        log_info = _parse_log(log_file)
        result_payload = _load_json(result_file)
        result_fresh = result_exists and result_payload is not None and result_mtime >= (log_mtime - 1.0)
        summary = {}
        record_count = 0
        success_count = 0
        failed_count = 0
        meta = {}
        result = {}
        if result_payload:
            meta = result_payload.get("meta", {}) if isinstance(result_payload, dict) else {}
            result = result_payload.get("result", {}) if isinstance(result_payload, dict) else {}
            if isinstance(result, dict):
                summary = result.get("summary", {}) or {}
            success_count = int(result.get("count", 0) or 0)
            failed_count = int(result.get("failed_count", 0) or 0)
            record_count = int(result.get("processed_count", success_count + failed_count) or 0)

        status = "running"
        if log_info.get("wrapper_exhausted"):
            status = "failed"
        elif log_info.get("wrapper_succeeded"):
            status = "finished"
        elif log_info["finished"] and result_payload is not None:
            status = "finished"
        if isinstance(result, dict):
            total_from_result = int(result.get("total_examples", 0) or 0)
            count_from_result = int(result.get("count", 0) or 0)
            if total_from_result > 0 and count_from_result >= total_from_result:
                status = "finished"
            elif status == "running" and total_from_result > 0 and count_from_result > 0:
                prog = log_info.get("latest_progress") or {}
                if isinstance(prog, dict):
                    done = int(prog.get("completed") or 0)
                    total_log = int(prog.get("total") or 0)
                    if total_log > 0 and done >= total_log:
                        status = "finished"

        # Detect stale sessions: no log activity for a long time while not finished/failed.
        STALE_SECONDS = 30 * 60  # 30 minutes
        log_age_s = time.time() - log_mtime if log_mtime else float("inf")
        if status == "running" and log_age_s > STALE_SECONDS:
            status = "failed"
            stale_reason = f"stale ({int(log_age_s // 60)}m inactive)"
        else:
            stale_reason = ""

        # A concise reason string used by the dashboard status column.
        status_reason = ""
        errs = log_info.get("error_lines") or []
        tail_str = " ".join(log_info.get("tail") or [])
        if status == "failed":
            if stale_reason:
                status_reason = stale_reason
            elif "502" in tail_str or any("502" in e for e in errs):
                status_reason = "upstream 502 streak"
            elif "exit_code=139" in tail_str:
                status_reason = "worker segfault (exit 139)"
            elif "exit_code=134" in tail_str:
                status_reason = "worker memory error (exit 134)"
            elif errs:
                status_reason = f"exception: {errs[-1][:120]}"
            else:
                status_reason = "retries exhausted"
        elif status == "finished":
            status_reason = "complete"
        else:
            status_reason = "running"

        progress = _normalize_progress(log_info.get("latest_progress"), status=status, record_count=record_count)
        progress_metrics = {}
        latest_progress = log_info.get("latest_progress")
        if isinstance(latest_progress, dict) and isinstance(latest_progress.get("metrics"), dict):
            progress_metrics = latest_progress.get("metrics") or {}
        same_run_result = False
        if meta.get("started_at") and log_info.get("started_at"):
            try:
                meta_ts = time.mktime(time.strptime(str(meta.get("started_at")), "%Y-%m-%d %H:%M:%S"))
                log_ts = time.mktime(time.strptime(str(log_info.get("started_at")), "%Y-%m-%d %H:%M:%S"))
                same_run_result = abs(meta_ts - log_ts) <= 15
            except Exception:
                same_run_result = str(meta.get("started_at")) == str(log_info.get("started_at"))

        # If the run is still in progress, prefer log progress over persisted
        # result files (which can be stale from previous attempts/runs).
        if status == "running":
            progress_completed = int(progress.get("completed", 0) or 0) if progress is not None else 0
            if result_payload is not None and (result_fresh or same_run_result):
                record_count = max(record_count, progress_completed)
            else:
                if progress is not None:
                    record_count = progress_completed
                elif isinstance(result, dict):
                    fallback_success = int(result.get("count", 0) or 0)
                    fallback_failed = int(result.get("failed_count", 0) or 0)
                    record_count = int(result.get("processed_count", fallback_success + fallback_failed) or 0)
                else:
                    record_count = 0

                # Keep available summary metrics when progress exists so active
                # sessions do not render blank metric cells in the ledger.
                if record_count <= 0:
                    summary = {}

            if progress_metrics:
                summary = {**summary, **progress_metrics}

        if status == "running" and progress is None and isinstance(result, dict):
            total_from_result = int(result.get("total_examples", 0) or 0)
            if total_from_result > 0:
                pct = (100.0 * record_count / total_from_result) if total_from_result else 0.0
                progress = {
                    "completed": record_count,
                    "total": total_from_result,
                    "pct": round(pct, 2),
                    "pct_text": f"{pct:.1f}%",
                }

        progress_ratio = _safe_float(progress.get("pct"), 0.0) / 100.0 if progress else 0.0

        score = _score(summary) if summary else 0.0
        metric_tiles = _metric_tiles(summary)
        supervision_profile = meta.get("dataset_profile") if isinstance(meta.get("dataset_profile"), dict) else log_info.get("supervision_profile")

        session = {
            "session_key": session_key,
            "dataset": dataset,
            "architecture": architecture,
            "status": status,
            "started_at": log_info["started_at"],
            "ended_at": meta.get("ended_at") or log_info["ended_at"],
            "records": record_count,
            "success_records": success_count,
            "failed_records": failed_count,
            "summary": summary,
            "metric_tiles": metric_tiles,
            "status_reason": status_reason,
            "score": round(score, 4),
            "result_file": str(result_file),
            "log_file": str(log_file),
            "meta": meta,
            "models": {
                "text": meta.get("text_model"),
                "vision": meta.get("vision_model"),
            },
            "thinking_budget": _extract_thinking_budget(meta),
            "duration_s": round(_safe_float(meta.get("duration_s"), 0.0), 3) if meta.get("duration_s") is not None else None,
            "supervision_profile": supervision_profile,
            "result_fresh": result_fresh,
            "progress": progress,
            "progress_ratio": round(progress_ratio, 4),
            "metric_digest": log_info.get("metric_digest"),
            "log_tail": log_info["tail"],
            "log_timeline": log_info["timeline"],
            "log_line_count": log_info["line_count"],
            "log_errors": log_info["error_lines"],
            "example_records": _extract_records_sample(result_payload),
            "progress_history": _extract_progress_history(log_info),
            "backbone": backbone_tag,
        }
        sessions.append(session)
        by_dataset[dataset].append(session)

    if not is_primary:
        return
    for ablation_json in sorted(ablation_results_dir.rglob("*.json")):
        payload = _load_json(ablation_json)
        if not payload:
            continue
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        ablation_tag = meta.get("ablation", "")
        if not ablation_tag:
            continue
        stem = ablation_json.stem
        session_key = f"ablation_{ablation_tag}_{stem}"
        if session_key in known_log_keys:
            continue
        dataset, architecture = _split_dataset_arch(session_key)
        result = payload.get("result", {}) if isinstance(payload, dict) else {}
        summary = result.get("summary", {}) or {} if isinstance(result, dict) else {}
        success_count = int(result.get("count", 0) or 0)
        failed_count = int(result.get("failed_count", 0) or 0)
        record_count = int(result.get("processed_count", success_count + failed_count) or 0)
        status = "finished" if record_count > 0 else "running"
        progress = _normalize_progress(None, status=status, record_count=record_count)
        progress_ratio = _safe_float(progress.get("pct", 0.0), 0.0) / 100.0 if progress else 0.0
        score = _score(summary) if summary else 0.0
        session = {
            "session_key": session_key,
            "dataset": dataset,
            "architecture": architecture,
            "status": status,
            "started_at": meta.get("started_at"),
            "ended_at": meta.get("ended_at"),
            "records": record_count,
            "success_records": success_count,
            "failed_records": failed_count,
            "summary": summary,
            "metric_tiles": _metric_tiles(summary),
            "status_reason": "complete" if status == "finished" else "running",
            "score": round(score, 4),
            "result_file": str(ablation_json),
            "log_file": "",
            "meta": meta,
            "models": {
                "text": meta.get("text_model"),
                "vision": meta.get("vision_model"),
            },
            "thinking_budget": _extract_thinking_budget(meta),
            "duration_s": round(_safe_float(meta.get("duration_s"), 0.0), 3) if meta.get("duration_s") is not None else None,
            "supervision_profile": meta.get("dataset_profile") if isinstance(meta.get("dataset_profile"), dict) else None,
            "result_fresh": True,
            "progress": progress,
            "progress_ratio": round(progress_ratio, 4),
            "metric_digest": None,
            "log_tail": [],
            "log_timeline": [],
            "log_line_count": 0,
            "log_errors": [],
            "example_records": _extract_records_sample(payload),
            "progress_history": None,
            "backbone": backbone_tag,
        }
        sessions.append(session)
        by_dataset[dataset].append(session)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dashboard JSON from experiment logs and result artifacts.")
    parser.add_argument("--results-dir", default="artifacts/experiments/results")
    parser.add_argument("--ablation-results-dir", default="artifacts/ablations")
    parser.add_argument("--logs-dir", default="logs/experiments/sessions")
    parser.add_argument("--primary-tag", default="qwen4b")
    parser.add_argument("--extra-logs-dir", default=None)
    parser.add_argument("--extra-results-dir", default=None)
    parser.add_argument("--extra-tag", default="qwen27b")
    parser.add_argument("--out", default="web/data/session_summary.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    ablation_results_dir = Path(args.ablation_results_dir)
    logs_dir = Path(args.logs_dir)
    if not results_dir.exists() and Path("artifacts/experiments").exists():
        results_dir = Path("artifacts/experiments")
    if not ablation_results_dir.exists():
        ablation_results_dir = Path("artifacts/ablations")
    if not logs_dir.exists() and Path("logs/experiments").exists():
        logs_dir = Path("logs/experiments")
    out_path = Path(args.out)

    sessions: list[dict[str, Any]] = []
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    known_log_keys: set[str] = set()

    _scan_track(
        logs_dir=logs_dir,
        results_dir=results_dir,
        ablation_results_dir=ablation_results_dir,
        backbone_tag=args.primary_tag,
        is_primary=True,
        sessions=sessions,
        by_dataset=by_dataset,
        known_log_keys=known_log_keys,
    )
    if args.extra_logs_dir:
        _scan_track(
            logs_dir=Path(args.extra_logs_dir),
            results_dir=Path(args.extra_results_dir) if args.extra_results_dir else results_dir,
            ablation_results_dir=ablation_results_dir,
            backbone_tag=args.extra_tag,
            is_primary=False,
            sessions=sessions,
            by_dataset=by_dataset,
            known_log_keys=known_log_keys,
        )

    _append_agentic_rag_sessions(root_dir=Path("."), sessions=sessions, by_dataset=by_dataset)
    _append_toolcall_cache_sessions(root_dir=Path("."), sessions=sessions, by_dataset=by_dataset)

    dataset_cards: list[dict[str, Any]] = []
    for dataset, rows in sorted(by_dataset.items()):
        finished = [item for item in rows if item["status"] == "finished"]
        best = max(finished, key=lambda item: item["score"]) if finished else None
        status_breakdown = {
            "finished": sum(1 for item in rows if item["status"] == "finished"),
            "running": sum(1 for item in rows if item["status"] == "running"),
            "failed": sum(1 for item in rows if item["status"] == "failed"),
        }
        avg_score = round(sum(item["score"] for item in finished) / len(finished), 4) if finished else None
        progress_values = [
            _safe_float(item.get("progress", {}).get("pct"), 0.0)
            for item in rows
            if isinstance(item.get("progress"), dict)
        ]
        avg_progress_pct = round(sum(progress_values) / len(progress_values), 2) if progress_values else None
        dataset_cards.append(
            {
                "dataset": dataset,
                "total_sessions": len(rows),
                "finished_sessions": len(finished),
                "best_architecture": best["architecture"] if best else None,
                "best_score": best["score"] if best else None,
                "best_metrics": best["metric_tiles"] if best else None,
                "average_score": avg_score,
                "average_progress_pct": avg_progress_pct,
                "status_breakdown": status_breakdown,
            }
        )

    sessions.sort(key=lambda item: (item["dataset"], -_safe_float(item.get("score"), 0.0), item["architecture"]))

    progress_sessions = [item for item in sessions if isinstance(item.get("progress"), dict)]
    total_examples = sum(int(item["progress"].get("total") or 0) for item in progress_sessions)
    completed_examples = sum(int(item["progress"].get("completed") or 0) for item in progress_sessions)
    overall_progress_pct = round((100.0 * completed_examples / total_examples), 2) if total_examples > 0 else 0.0

    payload = {
        "overview": {
            "total_sessions": len(sessions),
            "finished_sessions": sum(1 for item in sessions if item["status"] == "finished"),
            "running_sessions": sum(1 for item in sessions if item["status"] == "running"),
            "failed_sessions": sum(1 for item in sessions if item["status"] == "failed"),
            "overall_completed_examples": completed_examples,
            "overall_total_examples": total_examples,
            "overall_progress_pct": overall_progress_pct,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "generated_epoch": int(time.time()),
            "architecture_leaderboard": _architecture_leaderboard(sessions),
        },
        "dataset_cards": dataset_cards,
        "sessions": sessions,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(_scrub_nonfinite(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Wrote dashboard data to {out_path}")


def _scrub_nonfinite(obj):
    """Recursively replace NaN / Infinity with None so the emitted JSON is
    valid under RFC 8259 and can be parsed by ``JSON.parse`` in browsers."""
    import math
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _scrub_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub_nonfinite(v) for v in obj]
    return obj


if __name__ == "__main__":
    main()

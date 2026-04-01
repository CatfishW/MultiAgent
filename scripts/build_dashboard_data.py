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


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
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
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _parse_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = text.splitlines()

    timestamps = [match.group(1) for line in lines if (match := TIMESTAMP_RE.match(line))]
    started_at = timestamps[0] if timestamps else None
    ended_at = timestamps[-1] if timestamps else None

    finished = any("Finished session." in line for line in lines)
    error_lines = [line for line in lines if "Traceback" in line or "ERROR" in line or "Exception" in line]
    has_error = bool(error_lines)

    summary_line = next((line for line in reversed(lines) if "Summary:" in line), None)
    metric_digest = next((line.split("Metric digest:", 1)[1].strip() for line in reversed(lines) if "Metric digest:" in line), None)

    latest_progress = None
    for line in reversed(lines):
        match = PROGRESS_RE.search(line)
        if not match:
            continue
        latest_progress = {
            "completed": int(match.group("completed")),
            "total": int(match.group("total")),
            "pct_text": match.group("pct"),
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
        "token_f1": 0.28,
        "exact_match": 0.2,
        "rubric_coverage": 0.17,
        "grounded_overlap": 0.12,
        "citation_coverage": 0.08,
        "edu_json_compliance": 0.08,
        "edu_score_alignment": 0.07,
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

    if weight_total == 0.0:
        return 0.0
    return weighted_sum / weight_total


def _metric_tiles(summary: dict[str, Any]) -> dict[str, float]:
    keys = [
        "token_f1",
        "exact_match",
        "rubric_coverage",
        "grounded_overlap",
        "citation_coverage",
        "edu_json_compliance",
        "edu_score_alignment",
        "latency_ms",
        "agent_count",
    ]
    tiles: dict[str, float] = {}
    for key in keys:
        if key in summary:
            tiles[key] = round(_safe_float(summary.get(key), 0.0), 4)
    return tiles


def _extract_thinking_budget(meta: dict[str, Any]) -> int | None:
    text_extra = meta.get("text_chat_extra")
    if isinstance(text_extra, dict):
        value = text_extra.get("extra_body", {}).get("thinking_budget")
        if isinstance(value, int):
            return value
    vision_extra = meta.get("vision_chat_extra")
    if isinstance(vision_extra, dict):
        value = vision_extra.get("extra_body", {}).get("thinking_budget")
        if isinstance(value, int):
            return value
    return None


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dashboard JSON from experiment logs and result artifacts.")
    parser.add_argument("--results-dir", default="artifacts/experiments")
    parser.add_argument("--logs-dir", default="logs/experiments")
    parser.add_argument("--out", default="web/data/session_summary.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    logs_dir = Path(args.logs_dir)
    out_path = Path(args.out)

    sessions: list[dict[str, Any]] = []
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for log_file in sorted(logs_dir.glob("exp_*.log")):
        session_key = _session_name_to_key(log_file.name)
        dataset, architecture = _split_dataset_arch(session_key)
        result_file = results_dir / f"{session_key}.json"
        result_exists = result_file.exists()
        result_mtime = result_file.stat().st_mtime if result_exists else 0.0
        log_mtime = log_file.stat().st_mtime if log_file.exists() else 0.0

        log_info = _parse_log(log_file)
        result_payload = _load_json(result_file)
        result_fresh = result_exists and result_payload is not None and result_mtime >= (log_mtime - 1.0)
        summary = {}
        record_count = 0
        meta = {}
        if result_payload:
            meta = result_payload.get("meta", {}) if isinstance(result_payload, dict) else {}
            result = result_payload.get("result", {}) if isinstance(result_payload, dict) else {}
            if isinstance(result, dict):
                summary = result.get("summary", {}) or {}
                record_count = int(result.get("count", 0) or 0)

        status = "running"
        if log_info["has_error"]:
            status = "failed"
        elif log_info["finished"] and result_payload is not None:
            status = "finished"
        elif result_fresh and result_payload is not None:
            status = "finished"

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
            "summary": summary,
            "metric_tiles": metric_tiles,
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
            "progress": log_info.get("latest_progress"),
            "metric_digest": log_info.get("metric_digest"),
            "log_tail": log_info["tail"],
            "log_timeline": log_info["timeline"],
            "log_line_count": log_info["line_count"],
            "log_errors": log_info["error_lines"],
        }
        sessions.append(session)
        by_dataset[dataset].append(session)

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
        dataset_cards.append(
            {
                "dataset": dataset,
                "total_sessions": len(rows),
                "finished_sessions": len(finished),
                "best_architecture": best["architecture"] if best else None,
                "best_score": best["score"] if best else None,
                "best_metrics": best["metric_tiles"] if best else None,
                "average_score": avg_score,
                "status_breakdown": status_breakdown,
            }
        )

    sessions.sort(key=lambda item: (item["dataset"], -_safe_float(item.get("score"), 0.0), item["architecture"]))

    payload = {
        "overview": {
            "total_sessions": len(sessions),
            "finished_sessions": sum(1 for item in sessions if item["status"] == "finished"),
            "running_sessions": sum(1 for item in sessions if item["status"] == "running"),
            "failed_sessions": sum(1 for item in sessions if item["status"] == "failed"),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "architecture_leaderboard": _architecture_leaderboard(sessions),
        },
        "dataset_cards": dataset_cards,
        "sessions": sessions,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote dashboard data to {out_path}")


if __name__ == "__main__":
    main()

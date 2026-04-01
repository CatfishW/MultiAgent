from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _parse_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = text.splitlines()

    started_at = None
    for line in lines:
        match = re.match(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", line)
        if match:
            started_at = match.group(1)
            break

    finished = any("Finished session." in line for line in lines)
    has_error = "Traceback" in text or "Command exited with code" in text
    summary_line = next((line for line in reversed(lines) if "Summary:" in line), None)

    return {
        "started_at": started_at,
        "finished": finished,
        "has_error": has_error,
        "summary_line": summary_line,
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
    token_f1 = float(summary.get("token_f1", 0.0))
    exact = float(summary.get("exact_match", 0.0))
    grounded = float(summary.get("grounded_overlap", 0.0))
    return token_f1 * 0.55 + exact * 0.35 + grounded * 0.10


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

        log_info = _parse_log(log_file)
        result_payload = _load_json(result_file)
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
        if result_payload is not None:
            status = "finished"
        elif log_info["has_error"]:
            status = "failed"

        session = {
            "session_key": session_key,
            "dataset": dataset,
            "architecture": architecture,
            "status": status,
            "started_at": log_info["started_at"],
            "records": record_count,
            "summary": summary,
            "score": _score(summary) if summary else 0.0,
            "result_file": str(result_file),
            "log_file": str(log_file),
            "meta": meta,
            "log_tail": log_info["tail"],
        }
        sessions.append(session)
        by_dataset[dataset].append(session)

    dataset_cards: list[dict[str, Any]] = []
    for dataset, rows in sorted(by_dataset.items()):
        finished = [item for item in rows if item["status"] == "finished"]
        best = max(finished, key=lambda item: item["score"]) if finished else None
        dataset_cards.append(
            {
                "dataset": dataset,
                "total_sessions": len(rows),
                "finished_sessions": len(finished),
                "best_architecture": best["architecture"] if best else None,
                "best_score": best["score"] if best else None,
            }
        )

    payload = {
        "overview": {
            "total_sessions": len(sessions),
            "finished_sessions": sum(1 for item in sessions if item["status"] == "finished"),
            "running_sessions": sum(1 for item in sessions if item["status"] == "running"),
            "failed_sessions": sum(1 for item in sessions if item["status"] == "failed"),
        },
        "dataset_cards": dataset_cards,
        "sessions": sessions,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote dashboard data to {out_path}")


if __name__ == "__main__":
    main()

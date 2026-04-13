from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import time
from typing import Any

from huggingface_hub import list_repo_files, snapshot_download


DATASETS: dict[str, dict[str, str]] = {
    "EduBench": {
        "repo_id": "DirectionAI/EduBench",
        "snapshot_dir": "DirectionAI__EduBench",
        "attached_dir": "EduBench",
    },
    "TutorEval": {
        "repo_id": "princeton-nlp/TutorEval",
        "snapshot_dir": "princeton-nlp__TutorEval",
        "attached_dir": "TutorEval",
    },
}


def _iter_files(root: Path) -> set[str]:
    files: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        if rel.startswith(".cache/"):
            continue
        files.add(rel)
    return files


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def _dataset_stats(name: str, snapshot_dir: Path, remote_files: list[str]) -> dict[str, Any]:
    if name == "EduBench":
        by_file: dict[str, int] = {}
        total_rows = 0
        for rel in remote_files:
            if not rel.endswith(".jsonl"):
                continue
            count = _line_count(snapshot_dir / rel)
            by_file[rel] = count
            total_rows += count
        return {
            "total_rows": total_rows,
            "rows_by_file": by_file,
        }

    if name == "TutorEval":
        rows_by_file: dict[str, int] = {}
        try:
            import pyarrow.parquet as pq

            for rel in remote_files:
                if not rel.endswith(".parquet"):
                    continue
                table = pq.read_table(snapshot_dir / rel)
                rows_by_file[rel] = int(table.num_rows)
        except Exception:
            for rel in remote_files:
                if not rel.endswith(".parquet"):
                    continue
                rows_by_file[rel] = -1

        total_rows = sum(value for value in rows_by_file.values() if value >= 0)
        return {
            "total_rows": total_rows,
            "rows_by_file": rows_by_file,
        }

    return {}


def _sync_attached_folder(snapshot_dir: Path, attached_dir: Path, remote_files: list[str]) -> dict[str, int]:
    copied = 0
    updated = 0
    skipped = 0

    for rel in remote_files:
        src = snapshot_dir / rel
        dst = attached_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists():
            if dst.stat().st_size == src.stat().st_size:
                skipped += 1
                continue
            shutil.copy2(src, dst)
            updated += 1
            continue

        shutil.copy2(src, dst)
        copied += 1

    return {
        "copied": copied,
        "updated": updated,
        "skipped": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify full EduBench/TutorEval snapshots, then sync attached folders.")
    parser.add_argument("--datasets", nargs="+", default=["EduBench", "TutorEval"])
    parser.add_argument("--raw-root", default="data/raw/hf_datasets")
    parser.add_argument("--report", default="artifacts/dataset_audit/dataset_integrity_report.json")
    parser.add_argument("--skip-sync-attached-folders", action="store_true")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "raw_root": str(raw_root),
        "datasets": {},
    }

    for dataset_name in args.datasets:
        if dataset_name not in DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        spec = DATASETS[dataset_name]
        repo_id = spec["repo_id"]
        snapshot_dir = raw_root / spec["snapshot_dir"]
        attached_dir = Path(spec["attached_dir"])

        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(snapshot_dir),
        )

        remote_files = sorted(list_repo_files(repo_id, repo_type="dataset"))
        snapshot_files = _iter_files(snapshot_dir)

        missing_in_snapshot = sorted(set(remote_files) - snapshot_files)
        extra_in_snapshot = sorted(snapshot_files - set(remote_files))

        sync_result = {"copied": 0, "updated": 0, "skipped": 0}
        if not args.skip_sync_attached_folders:
            sync_result = _sync_attached_folder(snapshot_dir, attached_dir, remote_files)

        attached_remote_present = sum(1 for rel in remote_files if (attached_dir / rel).exists())
        attached_complete = attached_remote_present == len(remote_files)

        report["datasets"][dataset_name] = {
            "repo_id": repo_id,
            "snapshot_dir": str(snapshot_dir),
            "attached_dir": str(attached_dir),
            "remote_file_count": len(remote_files),
            "snapshot_file_count": len(snapshot_files),
            "missing_in_snapshot": missing_in_snapshot,
            "extra_in_snapshot": extra_in_snapshot,
            "attached_remote_present": attached_remote_present,
            "attached_complete": attached_complete,
            "sync_result": sync_result,
            "stats": _dataset_stats(dataset_name, snapshot_dir, remote_files),
        }

        print(f"[{dataset_name}] remote={len(remote_files)} snapshot={len(snapshot_files)} attached_complete={attached_complete}")
        print(f"[{dataset_name}] sync copied={sync_result['copied']} updated={sync_result['updated']} skipped={sync_result['skipped']}")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()

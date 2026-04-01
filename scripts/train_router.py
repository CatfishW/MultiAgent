from __future__ import annotations

import argparse
import json
from pathlib import Path

from eduagentic.app import ConferenceEduSystem


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the lightweight architecture router from JSONL data.")
    parser.add_argument("training_jsonl", help="Each line should contain {text,label}.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default="artifacts/router.pkl")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.training_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    texts = [str(row["text"]) for row in rows]
    labels = [str(row["label"]) for row in rows]
    system = ConferenceEduSystem(args.config)
    system.router.fit(texts, labels).save(args.out)
    print(f"Saved router to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()

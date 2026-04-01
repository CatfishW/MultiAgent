from __future__ import annotations

import argparse
from pathlib import Path

from eduagentic.app import ConferenceEduSystem


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local retrieval index from documents.")
    parser.add_argument("docs_path", help="Directory or file containing source documents.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default="artifacts/index")
    args = parser.parse_args()

    system = ConferenceEduSystem(args.config)
    index = system.index_documents(args.docs_path)
    out_path = index.save(args.out)
    print(f"Saved index to {Path(out_path).resolve()}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eduagentic.core.contracts import RetrievedChunk
from eduagentic.retrieval.reranker import LightweightReranker


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the lightweight reranker from JSONL pairs.")
    parser.add_argument("training_jsonl", help="Each line should contain {query,text,title,label,base_score,doc_id?,chunk_id?}.")
    parser.add_argument("--out", default="artifacts/reranker.pkl")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.training_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    queries = []
    chunks = []
    labels = []
    for idx, row in enumerate(rows):
        queries.append(str(row["query"]))
        chunks.append(
            RetrievedChunk(
                chunk_id=str(row.get("chunk_id", idx)),
                doc_id=str(row.get("doc_id", idx)),
                title=str(row.get("title", "document")),
                text=str(row["text"]),
                score=float(row.get("base_score", 0.0)),
                metadata={},
            )
        )
        labels.append(int(row["label"]))
    reranker = LightweightReranker().fit(queries, chunks, labels)
    reranker.save(args.out)
    print(f"Saved reranker to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()

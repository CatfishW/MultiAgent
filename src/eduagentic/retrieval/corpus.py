from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import json

from ..utils.text import chunk_tokens, stable_hash


@dataclass(slots=True)
class SourceDocument:
    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TextChunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    position: int
    metadata: dict[str, Any] = field(default_factory=dict)


KNOWN_TEXT_SUFFIXES = {".txt", ".md", ".jsonl", ".json"}


def _row_to_document(row: dict[str, Any], default_id_prefix: str) -> SourceDocument:
    doc_id = str(row.get("id") or row.get("doc_id") or stable_hash(default_id_prefix, row.get("title"), row.get("text")))
    title = str(row.get("title") or row.get("name") or row.get("topic") or doc_id)
    text = str(
        row.get("text")
        or row.get("content")
        or row.get("passage")
        or row.get("body")
        or row.get("document")
        or ""
    )
    metadata = {key: value for key, value in row.items() if key not in {"id", "doc_id", "title", "name", "topic", "text", "content", "passage", "body", "document"}}
    return SourceDocument(doc_id=doc_id, title=title, text=text, metadata=metadata)


def load_documents_from_path(path: str | Path) -> list[SourceDocument]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(root)
    paths = [root] if root.is_file() else sorted([p for p in root.rglob("*") if p.suffix.lower() in KNOWN_TEXT_SUFFIXES])
    documents: list[SourceDocument] = []
    for file_path in paths:
        suffix = file_path.suffix.lower()
        if suffix in {".txt", ".md"}:
            documents.append(SourceDocument(doc_id=stable_hash(file_path), title=file_path.stem, text=file_path.read_text(encoding="utf-8"), metadata={"path": str(file_path)}))
        elif suffix == ".jsonl":
            for idx, line in enumerate(file_path.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                documents.append(_row_to_document(json.loads(line), default_id_prefix=f"{file_path}:{idx}"))
        elif suffix == ".json":
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                for idx, row in enumerate(payload):
                    documents.append(_row_to_document(row, default_id_prefix=f"{file_path}:{idx}"))
            elif isinstance(payload, dict):
                if isinstance(payload.get("documents"), list):
                    for idx, row in enumerate(payload["documents"]):
                        documents.append(_row_to_document(row, default_id_prefix=f"{file_path}:{idx}"))
                else:
                    documents.append(_row_to_document(payload, default_id_prefix=str(file_path)))
    return documents


def chunk_documents(
    documents: Iterable[SourceDocument],
    *,
    chunk_size: int = 220,
    chunk_overlap: int = 40,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for document in documents:
        for position, text in enumerate(chunk_tokens(document.text, chunk_size=chunk_size, overlap=chunk_overlap)):
            chunk_id = stable_hash(document.doc_id, position, text[:80])
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    title=document.title,
                    text=text,
                    position=position,
                    metadata=dict(document.metadata),
                )
            )
    return chunks

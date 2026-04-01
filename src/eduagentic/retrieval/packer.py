from __future__ import annotations

from typing import Iterable

from ..core.contracts import RetrievedChunk
from ..utils.text import sentence_overlap, tokenize


class ContextPacker:
    def __init__(self, max_chars: int = 6000, mmr_lambda: float = 0.68) -> None:
        self.max_chars = max_chars
        self.mmr_lambda = mmr_lambda

    def select(self, query: str, chunks: Iterable[RetrievedChunk], final_k: int = 4) -> list[RetrievedChunk]:
        candidates = list(chunks)
        if not candidates:
            return []
        selected: list[RetrievedChunk] = []
        total_chars = 0
        while candidates and len(selected) < final_k:
            best_idx = 0
            best_value = float("-inf")
            for idx, chunk in enumerate(candidates):
                relevance = chunk.score
                redundancy = 0.0
                if selected:
                    redundancy = max(sentence_overlap(chunk.text, other.text) for other in selected)
                mmr = self.mmr_lambda * relevance - (1.0 - self.mmr_lambda) * redundancy
                if mmr > best_value:
                    best_value = mmr
                    best_idx = idx
            choice = candidates.pop(best_idx)
            projected = total_chars + len(choice.text)
            if selected and projected > self.max_chars:
                continue
            selected.append(choice)
            total_chars += len(choice.text)
        return selected

    def render_context(self, chunks: Iterable[RetrievedChunk]) -> str:
        parts: list[str] = []
        for chunk in chunks:
            parts.append(f"[{chunk.doc_id}:{chunk.chunk_id[:8]}] {chunk.title}\n{chunk.text}")
        return "\n\n".join(parts)

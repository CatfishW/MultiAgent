from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..core.contracts import RetrievedChunk
from ..utils.text import normalize_text, tokenize


@dataclass(slots=True)
class PairFeatures:
    base_score: float
    token_overlap: float
    title_overlap: float
    phrase_match: float
    numeric_overlap: float
    brevity: float

    def as_vector(self) -> list[float]:
        return [
            self.base_score,
            self.token_overlap,
            self.title_overlap,
            self.phrase_match,
            self.numeric_overlap,
            self.brevity,
        ]


class LightweightReranker:
    """Small trainable reranker.

    Default mode is a weighted heuristic. If `fit` is called, a logistic
    regression head replaces the manual combiner while staying lightweight.
    """

    def __init__(self) -> None:
        self.model: LogisticRegression | None = None

    def features(self, query: str, chunk: RetrievedChunk) -> PairFeatures:
        q_norm = normalize_text(query)
        c_norm = normalize_text(chunk.text)
        q_tokens = set(tokenize(query))
        c_tokens = set(tokenize(chunk.text))
        title_tokens = set(tokenize(chunk.title))
        token_overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens | c_tokens))
        title_overlap = len(q_tokens & title_tokens) / max(1, len(q_tokens | title_tokens))
        phrase_match = 1.0 if q_norm and q_norm in c_norm else 0.0
        q_numbers = {token for token in q_tokens if token.isdigit()}
        c_numbers = {token for token in c_tokens if token.isdigit()}
        numeric_overlap = len(q_numbers & c_numbers) / max(1, len(q_numbers | c_numbers) or 1)
        brevity = 1.0 / max(1.0, len(c_tokens) / 120.0)
        return PairFeatures(
            base_score=chunk.score,
            token_overlap=token_overlap,
            title_overlap=title_overlap,
            phrase_match=phrase_match,
            numeric_overlap=numeric_overlap,
            brevity=brevity,
        )

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        matrix = np.array([self.features(query, chunk).as_vector() for chunk in chunks])
        if self.model is not None:
            scores = self.model.predict_proba(matrix)[:, 1]
        else:
            weights = np.array([0.55, 0.20, 0.10, 0.06, 0.05, 0.04])
            scores = matrix @ weights
        order = np.argsort(scores)[::-1]
        reranked = []
        for idx in order:
            chunk = chunks[idx]
            reranked.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    text=chunk.text,
                    score=float(scores[idx]),
                    metadata=dict(chunk.metadata),
                )
            )
        return reranked

    def fit(self, queries: list[str], chunks: list[RetrievedChunk], labels: list[int]) -> "LightweightReranker":
        if not (len(queries) == len(chunks) == len(labels)):
            raise ValueError("queries, chunks, and labels must have the same length")
        X = np.array([self.features(query, chunk).as_vector() for query, chunk in zip(queries, chunks)])
        y = np.array(labels)
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.model.fit(X, y)
        return self

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as fh:
            pickle.dump(self, fh)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "LightweightReranker":
        with Path(path).open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, LightweightReranker):
            raise TypeError(f"Unexpected object type in reranker file: {type(obj)!r}")
        return obj

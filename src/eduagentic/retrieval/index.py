from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import pickle

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize

from ..config import RetrieverConfig
from ..core.contracts import RetrievedChunk, RetrievalResult
from ..utils.text import stable_hash, tokenize
from .corpus import TextChunk


class HybridIndex:
    """A lightweight hybrid retriever.

    It combines:
    - word-level TF-IDF for exact / phrase evidence
    - character TF-IDF for typo robustness
    - a compact latent projection for semantic smoothing
    """

    def __init__(self, config: RetrieverConfig | None = None) -> None:
        self.config = config or RetrieverConfig()
        self.chunks: list[TextChunk] = []
        self.word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=self.config.max_features,
            sublinear_tf=True,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=max(12000, self.config.max_features // 2),
            sublinear_tf=True,
        )
        self.word_matrix = None
        self.char_matrix = None
        self.svd: TruncatedSVD | None = None
        self.latent_matrix: np.ndarray | None = None
        self._fitted = False

    def fit(self, chunks: list[TextChunk]) -> "HybridIndex":
        self.chunks = list(chunks)
        texts = [self._surface_text(chunk) for chunk in self.chunks]
        self.word_matrix = self.word_vectorizer.fit_transform(texts)
        self.char_matrix = self.char_vectorizer.fit_transform(texts)
        self.svd = None
        self.latent_matrix = None
        if self.word_matrix.shape[0] >= 3 and self.word_matrix.shape[1] >= 8:
            n_components = min(
                self.config.latent_dim,
                self.word_matrix.shape[0] - 1,
                self.word_matrix.shape[1] - 1,
            )
            if n_components >= 2:
                self.svd = TruncatedSVD(n_components=n_components, random_state=42)
                self.latent_matrix = normalize(self.svd.fit_transform(self.word_matrix))
        self._fitted = True
        return self

    def _surface_text(self, chunk: TextChunk) -> str:
        title = chunk.title or ""
        return f"{title}\n{chunk.text}"

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("HybridIndex.fit must be called before search")

    def search(self, query: str, top_k: int | None = None) -> RetrievalResult:
        self._require_fitted()
        top_k = top_k or self.config.top_k
        query_word = self.word_vectorizer.transform([query])
        lexical = linear_kernel(query_word, self.word_matrix).ravel()
        query_char = self.char_vectorizer.transform([query])
        char_scores = linear_kernel(query_char, self.char_matrix).ravel()
        if self.svd is not None and self.latent_matrix is not None:
            latent_query = normalize(self.svd.transform(query_word))
            latent_scores = np.dot(self.latent_matrix, latent_query.T).ravel()
        else:
            latent_scores = np.zeros_like(lexical)

        scores = (
            self.config.lexical_weight * lexical
            + self.config.char_weight * char_scores
            + self.config.latent_weight * latent_scores
        )
        scores += self._apply_query_boosts(query)

        candidate_count = min(max(top_k, self.config.final_k), len(self.chunks))
        if candidate_count == 0:
            return RetrievalResult(query=query, chunks=[])
        if candidate_count == len(self.chunks):
            indices = np.argsort(scores)[::-1]
        else:
            partition = np.argpartition(scores, -candidate_count)[-candidate_count:]
            indices = partition[np.argsort(scores[partition])[::-1]]
        chunks = [
            RetrievedChunk(
                chunk_id=self.chunks[idx].chunk_id,
                doc_id=self.chunks[idx].doc_id,
                title=self.chunks[idx].title,
                text=self.chunks[idx].text,
                score=float(scores[idx]),
                metadata=dict(self.chunks[idx].metadata),
            )
            for idx in indices[:candidate_count]
        ]
        return RetrievalResult(query=query, chunks=chunks)

    def _apply_query_boosts(self, query: str) -> np.ndarray:
        query_tokens = set(tokenize(query))
        boosts = np.zeros(len(self.chunks), dtype=float)
        if not query_tokens:
            return boosts
        citation_mode = bool(query_tokens & {"cite", "citation", "evidence", "support", "reference"})
        for idx, chunk in enumerate(self.chunks):
            title_tokens = set(tokenize(chunk.title))
            overlap = len(query_tokens & title_tokens)
            boosts[idx] += 0.01 * overlap
            if citation_mode and chunk.metadata.get("source_type") in {"reference", "lecture", "document"}:
                boosts[idx] += 0.025
        return boosts

    def save(self, directory: str | Path) -> Path:
        self._require_fitted()
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        with (target / "hybrid_index.pkl").open("wb") as fh:
            pickle.dump(self, fh)
        return target / "hybrid_index.pkl"

    @classmethod
    def load(cls, path: str | Path) -> "HybridIndex":
        with Path(path).open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, HybridIndex):
            raise TypeError(f"Unexpected object in index file: {type(obj)!r}")
        return obj

    def fingerprint(self) -> str:
        joined = [chunk.chunk_id for chunk in self.chunks]
        return stable_hash(self.config, *joined)

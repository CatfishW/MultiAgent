from __future__ import annotations

from hashlib import sha256
import re

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_WS_RE = re.compile(r"\s+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.strip().lower()
    text = _WS_RE.sub(" ", text)
    return text


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _WORD_RE.findall(text or "")]


def stable_hash(*parts: object) -> str:
    joined = "||".join(str(part) for part in parts)
    return sha256(joined.encode("utf-8")).hexdigest()


def split_sentences(text: str) -> list[str]:
    stripped = (text or "").strip()
    if not stripped:
        return []
    return [piece.strip() for piece in _SENTENCE_RE.split(stripped) if piece.strip()]


def chunk_tokens(text: str, chunk_size: int = 220, overlap: int = 40) -> list[str]:
    words = (text or "").split()
    if not words:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def keyword_score(text: str, keywords: list[str]) -> float:
    lowered = normalize_text(text)
    if not keywords:
        return 0.0
    return sum(1.0 for keyword in keywords if keyword in lowered) / float(len(keywords))


def sentence_overlap(a: str, b: str) -> float:
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

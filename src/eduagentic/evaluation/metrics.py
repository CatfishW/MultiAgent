from __future__ import annotations

from statistics import mean
import re

from ..core.contracts import BenchmarkExample, PipelineResponse
from ..utils.text import normalize_text, tokenize


_OPTION_RE = re.compile(r"\(([A-Z])\)|\b([A-Z])\b")


def exact_match(prediction: str | None, gold: str | None) -> float:
    if prediction is None or gold is None:
        return 0.0
    return float(normalize_text(prediction) == normalize_text(gold))


def token_f1(prediction: str | None, gold: str | None) -> float:
    pred_tokens = tokenize(prediction or "")
    gold_tokens = tokenize(gold or "")
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = {token: pred_tokens.count(token) for token in set(pred_tokens)}
    gold_counts = {token: gold_tokens.count(token) for token in set(gold_tokens)}
    common = sum(min(pred_counts.get(token, 0), gold_counts.get(token, 0)) for token in set(pred_counts) | set(gold_counts))
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / max(precision + recall, 1e-8)


def choice_accuracy(prediction: str | None, choices: list[str] | None, gold: str | None) -> float:
    if not choices or gold is None or prediction is None:
        return 0.0
    pred_norm = normalize_text(prediction)
    gold_norm = normalize_text(gold)
    if pred_norm == gold_norm or gold_norm in pred_norm:
        return 1.0
    match = _OPTION_RE.search(prediction)
    if match:
        letter = match.group(1) or match.group(2)
        index = ord(letter) - ord("A")
        if 0 <= index < len(choices):
            return float(normalize_text(choices[index]) == gold_norm)
    return 0.0


def citation_coverage(response: PipelineResponse) -> float:
    if not response.retrieved_chunks:
        return 1.0 if not response.citations else 0.0
    cited = set(response.citations)
    retrieved = {chunk.doc_id for chunk in response.retrieved_chunks}
    if not retrieved:
        return 0.0
    return len(cited & retrieved) / len(retrieved)


def retrieval_doc_recall(response: PipelineResponse, expected_doc_ids: list[str]) -> float:
    if not expected_doc_ids:
        return 0.0
    retrieved = {chunk.doc_id for chunk in response.retrieved_chunks}
    expected = set(expected_doc_ids)
    return len(retrieved & expected) / len(expected)


def rubric_coverage(answer: str, rubric: list[str] | None) -> float:
    if not rubric:
        return 0.0
    lowered = normalize_text(answer)
    hits = 0
    for item in rubric:
        item_norm = normalize_text(item)
        if item_norm and item_norm[:40] in lowered:
            hits += 1
    return hits / len(rubric)


def grounded_overlap(answer: str, response: PipelineResponse) -> float:
    if not response.retrieved_chunks:
        return 0.0
    answer_tokens = set(tokenize(answer))
    if not answer_tokens:
        return 0.0
    chunk_tokens = set()
    for chunk in response.retrieved_chunks:
        chunk_tokens.update(tokenize(chunk.text))
    return len(answer_tokens & chunk_tokens) / max(1, len(answer_tokens))


def adaptivity_signal(example: BenchmarkExample, answer: str) -> float:
    if not example.dialogue_history:
        return 0.0
    lowered = normalize_text(answer)
    score = 0.0
    if "next step" in lowered or "try this" in lowered or "check" in lowered:
        score += 0.5
    if "because" in lowered or "common mistake" in lowered or "misconception" in lowered:
        score += 0.5
    return min(1.0, score)


def compute_metrics(example: BenchmarkExample, response: PipelineResponse) -> dict[str, float]:
    answer = response.answer or ""
    metrics = {
        "exact_match": exact_match(answer, example.gold_answer),
        "token_f1": token_f1(answer, example.gold_answer),
        "choice_accuracy": choice_accuracy(answer, example.choices, example.gold_answer),
        "citation_coverage": citation_coverage(response),
        "grounded_overlap": grounded_overlap(answer, response),
        "rubric_coverage": rubric_coverage(answer, example.rubric),
        "adaptivity": adaptivity_signal(example, answer),
        "retrieval_doc_recall": retrieval_doc_recall(response, example.expected_doc_ids),
        "latency_ms": float(response.metrics.get("latency_ms", 0.0)),
        "agent_count": float(response.metrics.get("agent_count", 0.0)),
    }
    return metrics


def summarize(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    keys = sorted(records[0].keys())
    return {key: mean(record[key] for record in records) for key in keys}

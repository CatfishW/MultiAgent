from __future__ import annotations

import ast
import json
from statistics import mean
import re
from typing import Any

from ..core.contracts import BenchmarkExample, PipelineResponse
from ..utils.text import normalize_text, tokenize


_OPTION_RE = re.compile(r"\(([A-Z])\)|\b([A-Z])\b")
_SCORE_PATTERN = re.compile(r'"?score"?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)', re.IGNORECASE)


def exact_match(prediction: str | None, gold: str | None) -> float:
    if prediction is None or gold is None:
        return 0.0
    return float(normalize_text(prediction) == normalize_text(gold))


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fenced_match:
        stripped = fenced_match.group(1).strip()

    candidates = [stripped]
    if "{" in stripped and "}" in stripped:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < end:
            sliced = stripped[start : end + 1]
            if sliced != stripped:
                candidates.append(sliced)

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _extract_score(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(max(0.0, min(100.0, float(value))))
    if isinstance(value, str):
        payload = _extract_json_object(value)
        if isinstance(payload, dict):
            nested = _extract_score(payload.get("Score", payload.get("score")))
            if nested is not None:
                return nested
        match = _SCORE_PATTERN.search(value)
        if match:
            return float(max(0.0, min(100.0, float(match.group(1)))))
    return None


def _coerce_text_from_payload(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first, dict) else None
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("content")
                        if isinstance(text, str) and text.strip():
                            parts.append(text.strip())
                if parts:
                    return "\n".join(parts)
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                return reasoning.strip()

    for key in ["output_text", "content", "text"]:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def canonical_answer_text(answer: str | None) -> str:
    if not answer:
        return ""
    text = answer.strip()
    if not text:
        return ""

    payload = _extract_json_object(text)
    if payload is None and text.startswith("{") and text.endswith("}"):
        try:
            maybe_payload = ast.literal_eval(text)
            if isinstance(maybe_payload, dict):
                payload = maybe_payload
        except Exception:
            payload = None

    if isinstance(payload, dict):
        extracted = _coerce_text_from_payload(payload)
        if extracted:
            return extracted

    return text


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
    answer_tokens = set(tokenize(lowered))
    if not answer_tokens:
        return 0.0
    hits = 0
    for item in rubric:
        item_norm = normalize_text(item)
        if not item_norm:
            continue
        if item_norm[:40] in lowered:
            hits += 1
            continue
        item_tokens = set(tokenize(item_norm))
        if not item_tokens:
            continue
        overlap = len(answer_tokens & item_tokens) / len(item_tokens)
        if overlap >= 0.35:
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


def edu_json_compliance(answer: str) -> float:
    payload = _extract_json_object(answer)
    if not isinstance(payload, dict):
        return 0.0

    has_score = _extract_score(payload.get("score", payload.get("Score"))) is not None
    has_details = any(key in payload for key in ["Scoring_Details", "Scoring Details", "scoring_details", "scoring details"])
    has_feedback = any(key in payload for key in ["Personalized Feedback", "personalized_feedback", "feedback"])
    return (float(has_score) + float(has_details) + float(has_feedback)) / 3.0


def edu_score_alignment(answer: str, reference_score: float | None) -> float:
    if reference_score is None:
        return 0.0
    predicted = _extract_score(answer)
    if predicted is None:
        payload = _extract_json_object(answer)
        if isinstance(payload, dict):
            predicted = _extract_score(payload.get("score", payload.get("Score")))
    if predicted is None:
        return 0.0
    delta = abs(predicted - reference_score)
    return max(0.0, 1.0 - (delta / 100.0))


def compute_metrics(example: BenchmarkExample, response: PipelineResponse) -> dict[str, float]:
    answer = canonical_answer_text(response.answer or "")
    profile = str(example.metadata.get("evaluation_profile", "")).lower()
    reference_score_raw = example.metadata.get("edubench_reference_score_mean")
    reference_score = float(reference_score_raw) if isinstance(reference_score_raw, (int, float)) else None

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
        "edu_json_compliance": 0.0,
        "edu_score_alignment": 0.0,
        "supervision_available": float(bool(example.gold_answer or example.rubric or reference_score is not None)),
    }

    if profile == "edubench_consensus":
        metrics["edu_json_compliance"] = edu_json_compliance(answer)
        metrics["edu_score_alignment"] = edu_score_alignment(answer, reference_score)

    return metrics


def summarize(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    keys = sorted(records[0].keys())
    return {key: mean(record[key] for record in records) for key in keys}

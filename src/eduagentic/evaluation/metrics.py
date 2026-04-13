from __future__ import annotations

import ast
import json
from statistics import mean
import re
from typing import Any

from ..core.contracts import BenchmarkExample, PipelineResponse
from ..utils.text import normalize_text, split_sentences, tokenize


_OPTION_RE = re.compile(r"\(([A-Z])\)|\b([A-Z])\b")
_SCORE_PATTERN = re.compile(r'"?score"?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)', re.IGNORECASE)

_EDUBENCH_SCENARIO_DIMENSIONS = ["iftc", "rtc", "crsc", "sei"]
_EDUBENCH_FACTUAL_DIMENSIONS = ["bfa", "dka", "rpr", "eicp"]
_EDUBENCH_PEDAGOGICAL_DIMENSIONS = ["csi", "mgp", "pas", "hots"]

_SUPPORTIVE_MARKERS = [
    "great",
    "good job",
    "well done",
    "keep",
    "you can",
    "you are",
    "let's",
    "encourage",
    "effort",
]
_GUIDANCE_MARKERS = [
    "next step",
    "try",
    "consider",
    "practice",
    "review",
    "focus on",
    "for example",
    "break it down",
]
_NEGATIVE_TONE_MARKERS = [
    "stupid",
    "idiot",
    "dumb",
    "useless",
]
_REASONING_MARKERS = [
    "because",
    "therefore",
    "since",
    "thus",
    "first",
    "second",
    "step",
    "explain",
    "reason",
]
_ERROR_MARKERS = [
    "incorrect",
    "mistake",
    "not correct",
    "however",
    "actually",
    "false",
    "true",
]
_CORRECTION_MARKERS = [
    "correct answer",
    "the answer is",
    "should be",
    "instead",
    "right answer",
]
_SIMPLICITY_MARKERS = [
    "step",
    "simple",
    "clearly",
    "for example",
    "in short",
    "break",
]
_HIGHER_ORDER_MARKERS = [
    "why",
    "how",
    "what if",
    "compare",
    "analyze",
    "reflect",
    "strategy",
    "explain your thinking",
]


def exact_match(prediction: str | None, gold: str | None) -> float:
    if prediction is None or gold is None:
        return 0.0
    return float(normalize_text(prediction) == normalize_text(gold))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_reference_score(value: float | None) -> float:
    if value is None:
        return 0.0
    if value > 1.0:
        return _clamp01(value / 100.0)
    return _clamp01(value)


def _keyword_fraction(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    lowered = normalize_text(text)
    hits = sum(1 for keyword in keywords if keyword in lowered)
    return _clamp01(hits / len(keywords))


def _extract_student_answer(question: str) -> str:
    match = re.search(r"student'?s\s*answer\s*:\s*(.+?)(?:\n|$)", question, flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip()


def _scenario_element_integration(example: BenchmarkExample, answer: str) -> float:
    elements: list[str] = []
    info = example.metadata.get("information")
    if isinstance(info, dict):
        for key in ["Subject", "Level", "Question"]:
            value = info.get(key)
            if isinstance(value, str) and value.strip():
                elements.append(value.strip())

    student_answer = _extract_student_answer(example.question)
    if student_answer:
        elements.append(student_answer)

    if not elements:
        return 0.0

    hits = 0
    lowered = normalize_text(answer)
    for element in elements:
        norm = normalize_text(element)
        if not norm:
            continue
        if norm in lowered:
            hits += 1
            continue
        if token_f1(answer, element) >= 0.2:
            hits += 1
    return _clamp01(hits / len(elements))


def _reasoning_rigor(answer: str) -> float:
    marker_score = _keyword_fraction(answer, _REASONING_MARKERS)
    sentences = split_sentences(answer)
    structured = 1.0 if (len(sentences) >= 2 or any(token in normalize_text(answer) for token in ["1.", "2.", "step 1"])) else 0.0
    return _clamp01(0.7 * marker_score + 0.3 * structured)


def _clarity_signal(answer: str) -> float:
    sentences = split_sentences(answer)
    if not sentences:
        return 0.0
    token_count = len(tokenize(answer))
    avg_tokens = token_count / max(1, len(sentences))
    readability = 1.0 - min(1.0, abs(avg_tokens - 18.0) / 18.0)
    simplicity = _keyword_fraction(answer, _SIMPLICITY_MARKERS)
    inspiration = _keyword_fraction(answer, _SUPPORTIVE_MARKERS)
    return _clamp01(0.5 * readability + 0.3 * simplicity + 0.2 * inspiration)


def _personalization_signal(example: BenchmarkExample, answer: str) -> float:
    tokens = set(tokenize(answer))
    second_person = 1.0 if ({"you", "your", "yours"} & tokens) else 0.0
    student_answer = _extract_student_answer(example.question)
    student_specific = token_f1(answer, student_answer) if student_answer else 0.0
    adaptive = adaptivity_signal(example, answer)
    return _clamp01(0.4 * second_person + 0.3 * student_specific + 0.3 * adaptive)


def _higher_order_signal(answer: str) -> float:
    higher_markers = _keyword_fraction(answer, _HIGHER_ORDER_MARKERS)
    asks_question = 1.0 if "?" in answer else 0.0
    return _clamp01(0.6 * higher_markers + 0.4 * asks_question)


def edubench_12d_scores(example: BenchmarkExample, response: PipelineResponse, answer: str, reference_score: float | None) -> dict[str, float]:
    ref_unit = _normalize_reference_score(reference_score)
    json_compliance = edu_json_compliance(answer)
    score_alignment = edu_score_alignment(answer, reference_score)
    rubric_match = rubric_coverage(answer, example.rubric)
    question_match = token_f1(answer, example.question)
    context_match = context_overlap(answer, example.context_text)

    supportive = _keyword_fraction(answer, _SUPPORTIVE_MARKERS)
    guidance = _keyword_fraction(answer, _GUIDANCE_MARKERS)
    negative_tone = _keyword_fraction(answer, _NEGATIVE_TONE_MARKERS)

    iftc = _clamp01(0.6 * json_compliance + 0.4 * rubric_match)
    rtc = _clamp01(0.7 * supportive + 0.3 * (1.0 - negative_tone))
    crsc = _clamp01(0.5 * question_match + 0.3 * rubric_match + 0.2 * context_match)
    sei = _scenario_element_integration(example, answer)

    bfa = _clamp01(0.7 * score_alignment + 0.3 * max(question_match, ref_unit))
    dka = _clamp01(0.7 * rubric_match + 0.3 * max(context_match, ref_unit))
    rpr = _reasoning_rigor(answer)
    eicp = _clamp01(0.5 * _keyword_fraction(answer, _ERROR_MARKERS) + 0.5 * max(_keyword_fraction(answer, _CORRECTION_MARKERS), score_alignment))

    csi = _clarity_signal(answer)
    mgp = _clamp01(0.6 * supportive + 0.4 * guidance)
    pas = _personalization_signal(example, answer)
    hots = _higher_order_signal(answer)

    scenario_values = [iftc, rtc, crsc, sei]
    factual_values = [bfa, dka, rpr, eicp]
    pedagogical_values = [csi, mgp, pas, hots]
    all_values = scenario_values + factual_values + pedagogical_values

    return {
        "edubench_iftc": iftc,
        "edubench_rtc": rtc,
        "edubench_crsc": crsc,
        "edubench_sei": sei,
        "edubench_bfa": bfa,
        "edubench_dka": dka,
        "edubench_rpr": rpr,
        "edubench_eicp": eicp,
        "edubench_csi": csi,
        "edubench_mgp": mgp,
        "edubench_pas": pas,
        "edubench_hots": hots,
        "edubench_scenario_adaptation": mean(scenario_values),
        "edubench_factual_reasoning_accuracy": mean(factual_values),
        "edubench_pedagogical_application": mean(pedagogical_values),
        "edubench_12d_mean": mean(all_values),
    }


def tutoreval_keypoint_hit_rate(answer: str, key_points: list[str]) -> float:
    if not key_points:
        return 0.0

    answer_tokens = set(tokenize(answer))
    lowered_answer = normalize_text(answer)
    if not answer_tokens:
        return 0.0

    point_scores: list[float] = []
    for point in key_points:
        norm_point = normalize_text(point)
        point_tokens = set(tokenize(norm_point))
        if not point_tokens:
            continue

        if norm_point and norm_point in lowered_answer:
            point_scores.append(1.0)
            continue

        overlap = len(answer_tokens & point_tokens) / len(point_tokens)
        if overlap >= 0.8:
            point_scores.append(1.0)
        elif overlap >= 0.4:
            point_scores.append(overlap)
        else:
            point_scores.append(0.0)

    return _clamp01(mean(point_scores) if point_scores else 0.0)


def tutoreval_secondary_scores(example: BenchmarkExample, answer: str, key_points: list[str], keypoint_hit_rate: float, keypoint_recall: float) -> dict[str, float]:
    target = example.gold_answer or " ".join(key_points)
    target_match = token_f1(answer, target)
    question_match = token_f1(answer, example.question)
    closed_book = bool(example.metadata.get("closed_book") or example.metadata.get("tutoreval_closed_book"))

    if closed_book:
        relevance = _clamp01(0.8 * question_match + 0.2 * target_match)
    else:
        relevance = _clamp01(0.5 * question_match + 0.3 * context_overlap(answer, example.context_text) + 0.2 * keypoint_hit_rate)

    correctness = _clamp01(0.7 * keypoint_hit_rate + 0.3 * target_match)
    completeness = _clamp01(0.8 * keypoint_hit_rate + 0.2 * keypoint_recall)

    return {
        "tutoreval_keypoint_hit_rate": keypoint_hit_rate,
        "tutoreval_correctness": correctness,
        "tutoreval_completeness": completeness,
        "tutoreval_relevance": relevance,
    }


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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def keypoint_token_alignment(answer: str, key_points: list[str]) -> tuple[float, float]:
    if not key_points:
        return 0.0, 0.0
    answer_tokens = set(tokenize(answer))
    if not answer_tokens:
        return 0.0, 0.0
    key_tokens: set[str] = set()
    for item in key_points:
        key_tokens.update(tokenize(item))
    if not key_tokens:
        return 0.0, 0.0
    overlap = answer_tokens & key_tokens
    precision = len(overlap) / len(answer_tokens)
    recall = len(overlap) / len(key_tokens)
    return precision, recall


def context_overlap(answer: str, context_text: str | None) -> float:
    if not context_text:
        return 0.0
    answer_tokens = set(tokenize(answer))
    context_tokens = set(tokenize(context_text))
    if not answer_tokens or not context_tokens:
        return 0.0
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def tutoreval_chapter_grounding(example: BenchmarkExample, response: PipelineResponse, answer: str) -> float:
    expected_in_chapter = bool(example.metadata.get("answer_in_chapter") or example.metadata.get("tutoreval_answer_in_chapter"))
    if not expected_in_chapter:
        return 0.0
    if response.retrieved_chunks:
        return grounded_overlap(answer, response)
    return context_overlap(answer, example.context_text)


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
    key_points = example.metadata.get("tutoreval_key_points")
    if not isinstance(key_points, list):
        key_points = list(example.rubric or [])
    key_points = [str(item) for item in key_points if str(item).strip()]

    metrics = {
        "exact_match": exact_match(answer, example.gold_answer),
        "token_f1": token_f1(answer, example.gold_answer),
        "choice_accuracy": choice_accuracy(answer, example.choices, example.gold_answer),
        "citation_coverage": citation_coverage(response),
        "grounded_overlap": grounded_overlap(answer, response),
        "rubric_coverage": rubric_coverage(answer, example.rubric),
        "adaptivity": adaptivity_signal(example, answer),
        "retrieval_doc_recall": retrieval_doc_recall(response, example.expected_doc_ids),
        "latency_ms": _safe_float(response.metrics.get("latency_ms", 0.0)),
        "api_time_ms": _safe_float(response.metrics.get("api_time_ms", 0.0)),
        "non_api_time_ms": _safe_float(response.metrics.get("non_api_time_ms", 0.0)),
        "api_time_ratio": _safe_float(response.metrics.get("api_time_ratio", 0.0)),
        "agent_count": _safe_float(response.metrics.get("agent_count", 0.0)),
        "llm_call_count": _safe_float(response.metrics.get("llm_call_count", 0.0)),
        "prompt_tokens": _safe_float(response.metrics.get("prompt_tokens", 0.0)),
        "completion_tokens": _safe_float(response.metrics.get("completion_tokens", 0.0)),
        "total_tokens": _safe_float(response.metrics.get("total_tokens", 0.0)),
        "retrieval_query_count": _safe_float(response.metrics.get("retrieval_query_count", 0.0)),
        "complexity_units": _safe_float(response.metrics.get("complexity_units", 0.0)),
        "complexity_per_second": _safe_float(response.metrics.get("complexity_per_second", 0.0)),
        "edu_json_compliance": 0.0,
        "edu_score_alignment": 0.0,
        "edubench_iftc": 0.0,
        "edubench_rtc": 0.0,
        "edubench_crsc": 0.0,
        "edubench_sei": 0.0,
        "edubench_bfa": 0.0,
        "edubench_dka": 0.0,
        "edubench_rpr": 0.0,
        "edubench_eicp": 0.0,
        "edubench_csi": 0.0,
        "edubench_mgp": 0.0,
        "edubench_pas": 0.0,
        "edubench_hots": 0.0,
        "edubench_scenario_adaptation": 0.0,
        "edubench_factual_reasoning_accuracy": 0.0,
        "edubench_pedagogical_application": 0.0,
        "edubench_12d_mean": 0.0,
        "tutoreval_keypoint_precision": 0.0,
        "tutoreval_keypoint_recall": 0.0,
        "tutoreval_keypoint_hit_rate": 0.0,
        "tutoreval_correctness": 0.0,
        "tutoreval_completeness": 0.0,
        "tutoreval_relevance": 0.0,
        "tutoreval_chapter_grounding": 0.0,
        "supervision_available": float(bool(example.gold_answer or example.rubric or reference_score is not None)),
    }

    if profile == "edubench_consensus":
        metrics["edu_json_compliance"] = edu_json_compliance(answer)
        metrics["edu_score_alignment"] = edu_score_alignment(answer, reference_score)
        metrics.update(edubench_12d_scores(example, response, answer, reference_score))
    elif profile == "tutoreval_key_points":
        key_precision, key_recall = keypoint_token_alignment(answer, key_points)
        metrics["tutoreval_keypoint_precision"] = key_precision
        metrics["tutoreval_keypoint_recall"] = key_recall
        hit_rate = tutoreval_keypoint_hit_rate(answer, key_points)
        metrics.update(tutoreval_secondary_scores(example, answer, key_points, hit_rate, key_recall))
        metrics["tutoreval_chapter_grounding"] = tutoreval_chapter_grounding(example, response, answer)

    return metrics


def summarize(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    keys = sorted(records[0].keys())
    return {key: mean(record[key] for record in records) for key in keys}

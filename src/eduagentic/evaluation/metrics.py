from __future__ import annotations

import ast
import json
import math
from statistics import mean
import re
from typing import Any

from ..core.contracts import BenchmarkExample, PipelineResponse
from ..utils.text import normalize_text, split_sentences, tokenize

# Minimal English stopword filter used by rubric_coverage for rarity check.
_STOPWORDS_RARE: frozenset[str] = frozenset(
    "the a an and or but if then else when while of for to in on at by with from as is are was were be been being do does did have has had this that these those it its their his her our your my we you they he she them us i so not no yes can could should would may might will shall there here into over under about between among within without per via than also such which who whom whose what where why how".split()
)

_NAN = float("nan")


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


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
    if _is_nan(value):
        return _NAN
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


def _walk_string_leaves(node: Any, depth: int, max_depth: int) -> list[str]:
    out: list[str] = []
    if isinstance(node, str):
        s = node.strip()
        if s:
            out.append(s)
    elif isinstance(node, dict) and depth < max_depth:
        for v in node.values():
            out.extend(_walk_string_leaves(v, depth + 1, max_depth))
    elif isinstance(node, (list, tuple)) and depth < max_depth:
        for v in node:
            out.extend(_walk_string_leaves(v, depth + 1, max_depth))
    return out


def _scenario_element_integration(example: BenchmarkExample, answer: str) -> float:
    elements: list[str] = []
    info = example.metadata.get("information")
    if isinstance(info, dict):
        elements.extend(_walk_string_leaves(info, depth=0, max_depth=2))

    student_answer = _extract_student_answer(example.question)
    if student_answer:
        elements.append(student_answer)

    # dedupe while preserving order, cap at 12 leaves to keep signal focused.
    seen: set[str] = set()
    deduped: list[str] = []
    for el in elements:
        key = el.lower()
        if key in seen or len(el) < 3:
            continue
        seen.add(key)
        deduped.append(el)
        if len(deduped) >= 12:
            break
    elements = deduped

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
        # Softer curve: full credit at >=0.7, linear from 0.25..0.7, 0 below 0.25.
        if overlap >= 0.7:
            point_scores.append(1.0)
        elif overlap >= 0.25:
            point_scores.append((overlap - 0.25) / (0.7 - 0.25))
        else:
            point_scores.append(0.0)

    return _clamp01(mean(point_scores) if point_scores else 0.0)


def tutoreval_secondary_scores(example: BenchmarkExample, answer: str, key_points: list[str], keypoint_hit_rate: float, keypoint_recall: float) -> dict[str, float]:
    target = example.gold_answer or " ".join(key_points)
    target_match = token_f1(answer, target)
    question_match = token_f1(answer, example.question)
    closed_book = bool(example.metadata.get("closed_book") or example.metadata.get("tutoreval_closed_book"))

    if closed_book:
        # Rebalanced: avoid rewarding pure question parroting. Mix question match with keypoint hits.
        relevance = _clamp01(0.5 * question_match + 0.5 * keypoint_hit_rate)
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
    """Tightened rubric coverage.

    A rubric item counts as covered if EITHER:
      - the first 40 characters appear verbatim in the answer, OR
      - >=55% of the item's tokens appear in the answer AND at least one rare
        (>=4 chars, non-stopword) rubric token is present.

    Rewarding verbosity on short generic words is suppressed by the rare-token gate.
    """
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
        if item_norm[:40] and item_norm[:40] in lowered:
            hits += 1
            continue
        item_tokens = set(tokenize(item_norm))
        if not item_tokens:
            continue
        overlap = len(answer_tokens & item_tokens) / len(item_tokens)
        if overlap < 0.55:
            continue
        rare_tokens = {t for t in item_tokens if len(t) >= 4 and t not in _STOPWORDS_RARE}
        if rare_tokens and not (rare_tokens & answer_tokens):
            continue
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


def corpus_factuality(answer: str, corpus_index: Any | None) -> float:
    """Retrieval-agnostic grounding proxy.

    Scores each non-trivial answer sentence against the shared corpus index (the
    same ``HybridIndex`` used by retrieval pipelines) by issuing the sentence as
    the query and taking the best top-k similarity. The result is the clipped
    mean across sentences.

    This metric is deliberately independent of whether the pipeline invoked
    retrieval. Every architecture is scored against the same corpus, so it does
    not structurally reward retrieval-enabled systems the way ``grounded_overlap``
    does (which returns 0.0 whenever ``response.retrieved_chunks`` is empty).

    Returns ``NaN`` if no corpus index is available or the answer has no usable
    sentences, so ``summarize()`` excludes the example rather than dragging the
    mean toward zero.
    """
    if corpus_index is None:
        return _NAN
    if not answer or not str(answer).strip():
        return _NAN
    search = getattr(corpus_index, "search", None)
    if search is None:
        return _NAN
    sentences = [s.strip() for s in split_sentences(answer) if s and len(tokenize(s)) >= 6]
    if not sentences:
        return _NAN
    # Cap sentence count so long answers do not dominate latency.
    sentences = sentences[:12]
    scores: list[float] = []
    for sentence in sentences:
        try:
            result = search(sentence, top_k=3)
        except Exception:
            continue
        chunks = getattr(result, "chunks", None) or []
        if not chunks:
            continue
        best = 0.0
        for chunk in chunks:
            value = float(getattr(chunk, "score", 0.0) or 0.0)
            if value > best:
                best = value
        scores.append(_clamp01(best))
    if not scores:
        return _NAN
    return _clamp01(mean(scores))


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
    """Micro-averaged per-keypoint precision + union-based recall.

    Precision: for each key-point, precision_k = |answer_tokens ∩ point_tokens| / |answer_tokens ∩ any_keypoint_tokens or answer_tokens|,
    then mean over non-empty key points. Recall: fraction of union(key-point tokens) seen in answer.
    """
    if not key_points:
        return 0.0, 0.0
    answer_tokens = set(tokenize(answer))
    if not answer_tokens:
        return 0.0, 0.0
    union_key_tokens: set[str] = set()
    per_point_tokens: list[set[str]] = []
    for item in key_points:
        toks = set(tokenize(item))
        if toks:
            per_point_tokens.append(toks)
            union_key_tokens.update(toks)
    if not union_key_tokens:
        return 0.0, 0.0
    # Per-keypoint precision: of the tokens the answer "spent" on this keypoint,
    # how many actually landed on it? denominator guards against trivial overlap.
    precisions: list[float] = []
    for toks in per_point_tokens:
        hits = answer_tokens & toks
        if not hits:
            precisions.append(0.0)
            continue
        # precision_k = |hits| / |hits ∪ extras| where extras are answer tokens that overlap any keypoint but not this one.
        extras = (answer_tokens & union_key_tokens) - toks
        denom = len(hits) + len(extras)
        precisions.append(len(hits) / denom if denom else 0.0)
    precision = mean(precisions) if precisions else 0.0
    recall = len(answer_tokens & union_key_tokens) / len(union_key_tokens)
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
        # Not applicable to this example -- return NaN so summarize() skips it.
        return _NAN
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
        # Not gradeable -- return NaN so ungradeable rows are excluded from group means.
        return _NAN
    predicted = _extract_score(answer)
    if predicted is None:
        payload = _extract_json_object(answer)
        if isinstance(payload, dict):
            predicted = _extract_score(payload.get("score", payload.get("Score")))
    if predicted is None:
        return 0.0
    delta = abs(predicted - reference_score)
    return max(0.0, 1.0 - (delta / 100.0))


def compute_metrics(
    example: BenchmarkExample,
    response: PipelineResponse,
    *,
    corpus_index: Any | None = None,
) -> dict[str, float]:
    """Compute the full metric dict for one ``(example, response)`` pair.

    Adds ``corpus_factuality`` when a shared corpus index is supplied. Omitting
    ``corpus_index`` preserves the historical behavior (metric is NaN and skipped
    by ``summarize``), so all existing call sites stay backward compatible.
    """
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
        "corpus_factuality": corpus_factuality(answer, corpus_index),
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
    """Mean over records, union of keys, NaN-skip per key.

    Ungradeable rows mark metrics as NaN; those are excluded from that metric's mean
    so partial-supervision datasets don't have their group means dragged toward zero.
    A per-key denominator is emitted as ``<key>_n`` for dashboard transparency.
    """
    if not records:
        return {}
    keys: set[str] = set()
    for record in records:
        if isinstance(record, dict):
            keys.update(record.keys())
    summary: dict[str, float] = {}
    for key in sorted(keys):
        total = 0.0
        n = 0
        for record in records:
            if not isinstance(record, dict):
                continue
            if key not in record:
                continue
            value = record[key]
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if math.isnan(v):
                continue
            total += v
            n += 1
        summary[key] = (total / n) if n > 0 else _NAN
        summary[f"{key}_n"] = float(n)
    return summary

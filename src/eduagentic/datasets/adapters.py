from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable
import json

from ..core.contracts import BenchmarkExample, ConversationTurn
from ..utils.text import stable_hash
from .base import DatasetAdapter, DatasetSpec


TransformFn = Callable[[dict[str, Any], DatasetSpec], BenchmarkExample]


def _safe_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _parse_json_if_needed(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                return json.loads(stripped)
            except Exception:
                return value
    return value


def _first_present(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
    def _is_present(value: Any) -> bool:
        return value is not None and value != ""

    normalized = {
        str(key).strip().lower().replace(" ", "_"): value
        for key, value in row.items()
    }
    for key in keys:
        if key in row and _is_present(row[key]):
            return row[key]
        normalized_key = key.strip().lower().replace(" ", "_")
        if normalized_key in normalized and _is_present(normalized[normalized_key]):
            return normalized[normalized_key]
    return default


def generic_text_transform(row: dict[str, Any], spec: DatasetSpec) -> BenchmarkExample:
    question = str(_first_present(row, "question", "query", "prompt", "instruction", "input", default=""))
    answer = _first_present(row, "answer", "gold", "gold_answer", "target", "output", "label", "response", "corrected_answer")
    context = _first_present(row, "context", "passage", "document", "chapter", "lecture", "source", "history")
    rubric = _parse_json_if_needed(_first_present(row, "rubric", "criteria", "evaluation_criteria"))
    if rubric is None:
        rubric_list = []
    elif isinstance(rubric, list):
        rubric_list = [str(item) for item in rubric]
    else:
        rubric_list = [str(rubric)]
    example_id = str(_first_present(row, "id", "example_id", "uid", default=stable_hash(spec.name, question[:80], answer)))
    history = []
    history_payload = _parse_json_if_needed(_first_present(row, "dialogue", "history", "messages", default=[]))
    if isinstance(history_payload, list):
        for item in history_payload:
            if isinstance(item, dict):
                history.append(ConversationTurn(role=str(item.get("role", "user")), text=str(item.get("text") or item.get("content") or "")))
    images = _safe_list(_first_present(row, "images", "image", "image_path"))
    return BenchmarkExample(
        example_id=example_id,
        dataset_name=spec.name,
        regime_hint=spec.regime,
        question=question,
        gold_answer=str(answer) if answer is not None else None,
        context_text=str(context) if context is not None and not isinstance(context, list) else None,
        dialogue_history=history,
        rubric=rubric_list or None,
        images=[str(item) for item in images if item],
        metadata={key: value for key, value in row.items() if key not in {"question", "query", "prompt", "instruction", "input", "answer", "gold", "gold_answer", "target", "output", "label", "context", "passage", "document", "chapter", "lecture", "source", "history", "dialogue", "messages", "rubric", "criteria", "evaluation_criteria", "images", "image", "image_path"}},
    )


def hotpot_transform(row: dict[str, Any], spec: DatasetSpec) -> BenchmarkExample:
    question = str(row.get("question", ""))
    answer = row.get("answer")
    reference_docs = []
    context = row.get("context") or []
    if isinstance(context, list):
        for idx, item in enumerate(context):
            if isinstance(item, list) and len(item) == 2:
                title = str(item[0])
                body = " ".join(item[1]) if isinstance(item[1], list) else str(item[1])
                reference_docs.append({"id": f"doc-{idx}", "title": title, "text": body, "source_type": "reference"})
    supporting = row.get("supporting_facts") or []
    expected_doc_ids = []
    if isinstance(supporting, list):
        for item in supporting:
            if isinstance(item, list) and item:
                expected_doc_ids.append(str(item[0]))
    return BenchmarkExample(
        example_id=str(row.get("id", stable_hash(spec.name, question))),
        dataset_name=spec.name,
        regime_hint=spec.regime,
        question=question,
        gold_answer=str(answer) if answer is not None else None,
        reference_docs=reference_docs,
        expected_doc_ids=expected_doc_ids,
        metadata={"raw_type": "hotpot"},
    )


def fever_transform(row: dict[str, Any], spec: DatasetSpec) -> BenchmarkExample:
    claim = str(row.get("claim", row.get("question", "")))
    evidence_rows = row.get("evidence") or []
    docs = []
    for idx, item in enumerate(evidence_rows[:8]):
        docs.append({"id": f"evidence-{idx}", "title": f"evidence-{idx}", "text": str(item), "source_type": "reference"})
    label = row.get("label")
    return BenchmarkExample(
        example_id=str(row.get("id", stable_hash(spec.name, claim))),
        dataset_name=spec.name,
        regime_hint=spec.regime,
        question=claim,
        gold_answer=str(label) if label is not None else None,
        reference_docs=docs,
        expected_doc_ids=[item["id"] for item in docs],
    )


def scienceqa_transform(row: dict[str, Any], spec: DatasetSpec) -> BenchmarkExample:
    question = str(row.get("question", ""))
    choices = [str(choice) for choice in _safe_list(row.get("choices"))]
    answer = row.get("answer")
    if isinstance(answer, int) and 0 <= answer < len(choices):
        gold = choices[answer]
    else:
        gold = str(answer) if answer is not None else None
    lecture = row.get("lecture") or row.get("hint") or row.get("solution")
    images = _safe_list(row.get("image") or row.get("image_path"))
    return BenchmarkExample(
        example_id=str(row.get("id", stable_hash(spec.name, question))),
        dataset_name=spec.name,
        regime_hint=spec.regime,
        question=question,
        gold_answer=gold,
        choices=choices or None,
        context_text=str(lecture) if lecture else None,
        images=[str(item) for item in images if item],
        metadata={"raw_type": "scienceqa"},
    )


def wizard_transform(row: dict[str, Any], spec: DatasetSpec) -> BenchmarkExample:
    topic = row.get("chosen_topic") or row.get("topic")
    history_payload = row.get("dialog") or row.get("messages") or []
    history = []
    question = ""
    gold = None
    if isinstance(history_payload, list):
        for item in history_payload:
            if isinstance(item, dict):
                role = "assistant" if item.get("speaker", "").lower().startswith("wizard") else "user"
                text = str(item.get("text", ""))
                history.append(ConversationTurn(role=role, text=text))
        if history:
            question = history[-1].text if history[-1].role == "user" else (history[-2].text if len(history) > 1 else "")
            gold = history[-1].text if history[-1].role == "assistant" else None
    return BenchmarkExample(
        example_id=str(row.get("id", stable_hash(spec.name, topic, question))),
        dataset_name=spec.name,
        regime_hint=spec.regime,
        question=question or str(topic or "Discuss the topic."),
        gold_answer=gold,
        dialogue_history=history[:-1] if history and history[-1].role == "assistant" else history,
        metadata={"topic": topic},
    )


def long_context_transform(row: dict[str, Any], spec: DatasetSpec) -> BenchmarkExample:
    example = generic_text_transform(row, spec)
    if example.context_text is None:
        example.context_text = str(_first_present(row, "context", "article", "document", "passage", default=""))
    return example


TRANSFORMS: dict[str, TransformFn] = {
    "generic": generic_text_transform,
    "hotpot": hotpot_transform,
    "fever": fever_transform,
    "scienceqa": scienceqa_transform,
    "wizard": wizard_transform,
    "long_context": long_context_transform,
}


class LocalJsonlAdapter(DatasetAdapter):
    def __init__(self, spec: DatasetSpec, transform: TransformFn = generic_text_transform) -> None:
        super().__init__(spec)
        self.transform = transform

    def load(self, source: str | None = None, split: str | None = None, limit: int | None = None) -> list[BenchmarkExample]:
        if not source:
            raise FileNotFoundError(f"Dataset {self.spec.name} requires a local JSONL source path")
        path = Path(source)
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
        examples = [self.transform(row, self.spec) for row in rows]
        return examples[:limit] if limit is not None else examples


class LocalJsonAdapter(DatasetAdapter):
    def __init__(self, spec: DatasetSpec, transform: TransformFn = generic_text_transform) -> None:
        super().__init__(spec)
        self.transform = transform

    def load(self, source: str | None = None, split: str | None = None, limit: int | None = None) -> list[BenchmarkExample]:
        if not source:
            raise FileNotFoundError(f"Dataset {self.spec.name} requires a local JSON source path")
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
        if isinstance(payload, dict) and split and split in payload:
            rows = payload[split]
        elif isinstance(payload, dict) and "data" in payload:
            rows = payload["data"]
        else:
            rows = payload
        examples = [self.transform(row, self.spec) for row in rows]
        return examples[:limit] if limit is not None else examples


class HuggingFaceAdapter(DatasetAdapter):
    def __init__(self, spec: DatasetSpec, dataset_id: str, *, transform: TransformFn = generic_text_transform, subset: str | None = None) -> None:
        super().__init__(spec)
        self.dataset_id = dataset_id
        self.transform = transform
        self.subset = subset

    def load(self, source: str | None = None, split: str | None = None, limit: int | None = None) -> list[BenchmarkExample]:
        try:
            from datasets import load_dataset
        except Exception as exc:
            raise RuntimeError(
                "The optional 'datasets' package is required for Hugging Face-backed adapters. "
                "Install it or export the dataset to local JSONL and point the registry override there."
            ) from exc
        actual_id = source or self.dataset_id
        resolved_split = split or self.spec.split
        try:
            dataset = load_dataset(actual_id, name=self.subset, split=resolved_split)
            rows = list(dataset.select(range(min(limit, len(dataset))))) if limit is not None else list(dataset)
        except Exception:
            streamed = load_dataset(actual_id, name=self.subset, split=resolved_split, streaming=True)
            rows = []
            for idx, row in enumerate(streamed):
                rows.append(dict(row))
                if limit is not None and idx + 1 >= limit:
                    break
        return [self.transform(dict(row), self.spec) for row in rows]

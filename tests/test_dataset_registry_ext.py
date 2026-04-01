from __future__ import annotations

import json

from eduagentic.config import DEFAULT_CONFIG, load_app_config
from eduagentic.datasets.registry import DatasetRegistry


EXPECTED_DATASETS = {
    "EduBench",
    "TutorEval",
    "LM-Science-Tutor",
    "ScienceQA",
    "MathTutorBench",
    "TutorBench",
    "HotpotQA",
    "AgentBench",
    "SCROLLS",
    "LongBench-v2",
    "BEIR",
    "FEVER",
    "Wizard of Wikipedia",
}


def test_registry_covers_all_pdf_dataset_families():
    registry = DatasetRegistry(DEFAULT_CONFIG)
    assert EXPECTED_DATASETS.issubset(set(registry.names()))


def test_local_jsonl_adapter_loads_generic_examples(tmp_path):
    sample = {
        "id": "ex-1",
        "question": "What is photosynthesis?",
        "answer": "A process plants use to convert light into chemical energy.",
        "context": "Plants use sunlight, water, and carbon dioxide.",
    }
    path = tmp_path / "sample.jsonl"
    path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    config = load_app_config({"datasets": {"registry_overrides": {"EduBench": {"default_loader": "local_jsonl", "default_source": str(path)}}}})
    registry = DatasetRegistry(config)
    loaded = registry.load("EduBench")
    assert loaded[0].question == sample["question"]
    assert loaded[0].gold_answer == sample["answer"]

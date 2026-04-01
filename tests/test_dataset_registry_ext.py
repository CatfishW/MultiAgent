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


def test_edubench_transform_extracts_consensus_score_and_rubric(tmp_path):
    sample = {
        "id": "edu-1",
        "question": "Score this student answer and provide feedback.",
        "answer": None,
        "metadata": {
            "information": {"Subject": "Math"},
            "model_predictions": [
                {
                    "model": "m1",
                    "response": "{\"Score\": 70, \"Scoring_Details\": {\"accuracy\": 60}, \"Personalized Feedback\": \"More detail needed.\"}",
                },
                {
                    "model": "m2",
                    "response": "```json\n{\"Score\": 90, \"Scoring Details\": \"Good structure\", \"Personalized Feedback\": \"Nice work\"}\n```",
                },
            ],
        },
    }
    path = tmp_path / "edubench.jsonl"
    path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    config = load_app_config({"datasets": {"registry_overrides": {"EduBench": {"default_loader": "local_jsonl", "default_source": str(path)}}}})
    registry = DatasetRegistry(config)

    loaded = registry.load("EduBench")
    assert loaded[0].gold_answer is None
    assert loaded[0].metadata["evaluation_profile"] == "edubench_consensus"
    assert loaded[0].metadata["edubench_reference_score_mean"] == 80.0
    assert loaded[0].rubric is not None
    assert "Score" in loaded[0].rubric


def test_tutoreval_transform_uses_key_points_for_supervision(tmp_path):
    sample = {
        "id": "te-1",
        "question": "Which is the fastest mode of heat transfer?",
        "answer": None,
        "metadata": {
            "key_points": "- Answer is radiation\n- explain that it does not require a medium",
            "closed_book": True,
        },
    }
    path = tmp_path / "tutoreval.jsonl"
    path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    config = load_app_config({"datasets": {"registry_overrides": {"TutorEval": {"default_loader": "local_jsonl", "default_source": str(path)}}}})
    registry = DatasetRegistry(config)

    loaded = registry.load("TutorEval")
    assert loaded[0].metadata["evaluation_profile"] == "tutoreval_key_points"
    assert loaded[0].gold_answer is not None
    assert "radiation" in loaded[0].gold_answer.lower()
    assert loaded[0].rubric is not None
    assert any("radiation" in item.lower() for item in loaded[0].rubric)

from __future__ import annotations

from eduagentic.core.contracts import ArchitectureFamily, BenchmarkExample, PipelineResponse, RouteDecision, TaskRegime
from eduagentic.evaluation.metrics import canonical_answer_text, compute_metrics


def _route() -> RouteDecision:
    return RouteDecision(
        regime=TaskRegime.ADAPTIVE_TUTORING,
        architecture=ArchitectureFamily.HYBRID_FAST,
        require_retrieval=False,
        use_critic=False,
        use_rubric_agent=True,
    )


def test_canonical_answer_text_extracts_reasoning_content_from_payload_string() -> None:
    payload = {
        "id": "x",
        "choices": [
            {
                "message": {
                    "content": None,
                    "reasoning_content": "Final answer: Radiation is the fastest mode of heat transfer.",
                }
            }
        ],
    }
    extracted = canonical_answer_text(str(payload))
    assert "fastest mode of heat transfer" in extracted


def test_compute_metrics_reports_edubench_alignment_metrics() -> None:
    example = BenchmarkExample(
        example_id="edu-1",
        dataset_name="EduBench",
        regime_hint=TaskRegime.ADAPTIVE_TUTORING,
        question="Score the student answer",
        rubric=["Score", "Scoring Details", "Personalized Feedback"],
        metadata={
            "evaluation_profile": "edubench_consensus",
            "edubench_reference_score_mean": 80.0,
        },
    )
    response = PipelineResponse(
        answer='{"Score": 78, "Scoring_Details": {"Accuracy": 80}, "Personalized Feedback": "Good effort."}',
        architecture=ArchitectureFamily.HYBRID_FAST,
        regime=TaskRegime.ADAPTIVE_TUTORING,
        route=_route(),
        metrics={"latency_ms": 150.0, "agent_count": 4.0},
    )

    metrics = compute_metrics(example, response)

    assert metrics["edu_json_compliance"] >= 0.99
    assert metrics["edu_score_alignment"] > 0.95
    assert metrics["supervision_available"] == 1.0

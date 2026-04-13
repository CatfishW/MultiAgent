from __future__ import annotations

from eduagentic.core.contracts import ArchitectureFamily, BenchmarkExample, PipelineResponse, RetrievedChunk, RouteDecision, TaskRegime
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
        answer='{"Score": 78, "Scoring_Details": {"Accuracy": 80}, "Personalized Feedback": "Great effort. You can review this step by step and practice one more example. Why do you think this works?"}',
        architecture=ArchitectureFamily.HYBRID_FAST,
        regime=TaskRegime.ADAPTIVE_TUTORING,
        route=_route(),
        metrics={"latency_ms": 150.0, "agent_count": 4.0},
    )

    metrics = compute_metrics(example, response)

    assert metrics["edu_json_compliance"] >= 0.99
    assert metrics["edu_score_alignment"] > 0.95
    assert metrics["edubench_iftc"] > 0.7
    assert metrics["edubench_scenario_adaptation"] > 0.35
    assert metrics["edubench_factual_reasoning_accuracy"] > 0.4
    assert metrics["edubench_pedagogical_application"] > 0.2
    assert metrics["edubench_12d_mean"] > 0.3
    for key in [
        "edubench_rtc",
        "edubench_crsc",
        "edubench_sei",
        "edubench_bfa",
        "edubench_dka",
        "edubench_rpr",
        "edubench_eicp",
        "edubench_csi",
        "edubench_mgp",
        "edubench_pas",
        "edubench_hots",
    ]:
        assert 0.0 <= metrics[key] <= 1.0
    assert metrics["supervision_available"] == 1.0


def test_compute_metrics_includes_api_and_complexity_telemetry() -> None:
    example = BenchmarkExample(
        example_id="telemetry-1",
        dataset_name="TutorEval",
        regime_hint=TaskRegime.ADAPTIVE_TUTORING,
        question="What is conduction?",
        gold_answer="Conduction is heat transfer through direct contact.",
    )
    response = PipelineResponse(
        answer="Conduction is heat transfer through direct contact.",
        architecture=ArchitectureFamily.HYBRID_FAST,
        regime=TaskRegime.ADAPTIVE_TUTORING,
        route=_route(),
        metrics={
            "latency_ms": 120.0,
            "api_time_ms": 90.0,
            "non_api_time_ms": 30.0,
            "api_time_ratio": 0.75,
            "agent_count": 4.0,
            "llm_call_count": 2.0,
            "prompt_tokens": 120.0,
            "completion_tokens": 35.0,
            "total_tokens": 155.0,
            "retrieval_query_count": 2.0,
            "complexity_units": 980.0,
            "complexity_per_second": 8166.67,
        },
    )

    metrics = compute_metrics(example, response)

    assert metrics["api_time_ms"] == 90.0
    assert metrics["llm_call_count"] == 2.0
    assert metrics["total_tokens"] == 155.0
    assert metrics["complexity_units"] == 980.0
    assert metrics["complexity_per_second"] > 8000.0


def test_compute_metrics_reports_tutoreval_keypoint_metrics() -> None:
    example = BenchmarkExample(
        example_id="te-1",
        dataset_name="TutorEval",
        regime_hint=TaskRegime.ADAPTIVE_TUTORING,
        question="Which is the fastest mode of heat transfer?",
        context_text="Heat transfer modes include conduction, convection, and radiation. Radiation is the fastest.",
        metadata={
            "evaluation_profile": "tutoreval_key_points",
            "tutoreval_key_points": [
                "Radiation is the fastest mode of heat transfer.",
                "Conduction and convection are slower.",
            ],
            "answer_in_chapter": True,
        },
    )
    response = PipelineResponse(
        answer="Radiation is the fastest mode of heat transfer, while conduction and convection are slower.",
        architecture=ArchitectureFamily.HYBRID_FAST,
        regime=TaskRegime.ADAPTIVE_TUTORING,
        route=_route(),
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id="c1",
                doc_id="doc-1",
                title="Heat Transfer",
                text="Radiation is the fastest mode of heat transfer.",
                score=0.9,
            )
        ],
        metrics={"latency_ms": 140.0, "agent_count": 3.0},
    )

    metrics = compute_metrics(example, response)

    assert metrics["tutoreval_keypoint_precision"] > 0.4
    assert metrics["tutoreval_keypoint_recall"] > 0.4
    assert metrics["tutoreval_keypoint_hit_rate"] > 0.4
    assert metrics["tutoreval_correctness"] > 0.4
    assert metrics["tutoreval_completeness"] > 0.4
    assert metrics["tutoreval_relevance"] > 0.3
    assert metrics["tutoreval_chapter_grounding"] > 0.2

from __future__ import annotations

import math

import pytest

from eduagentic.core.contracts import (
    ArchitectureFamily,
    BenchmarkExample,
    Modality,
    PipelineResponse,
    RetrievedChunk,
    RouteDecision,
    TaskRegime,
)
from eduagentic.evaluation.metrics import (
    compute_metrics,
    edu_score_alignment,
    keypoint_token_alignment,
    rubric_coverage,
    summarize,
    tutoreval_chapter_grounding,
    tutoreval_keypoint_hit_rate,
)


def _example(**kwargs) -> BenchmarkExample:
    base = dict(
        example_id="ex",
        dataset_name="Test",
        regime_hint=None,
        question="What is photosynthesis?",
        gold_answer=None,
        choices=None,
        context_text=None,
        dialogue_history=[],
        rubric=None,
        images=None,
        metadata={},
        reference_docs=[],
        expected_doc_ids=[],
    )
    base.update(kwargs)
    return BenchmarkExample(**base)


def _response(answer: str, chunks: list[RetrievedChunk] | None = None) -> PipelineResponse:
    route = RouteDecision(
        regime=TaskRegime.EVIDENCE_GROUNDED,
        architecture=ArchitectureFamily.HYBRID_FAST,
        require_retrieval=False,
        use_critic=False,
        use_rubric_agent=False,
        modality=Modality.TEXT,
    )
    return PipelineResponse(
        answer=answer,
        architecture=ArchitectureFamily.HYBRID_FAST,
        regime=TaskRegime.EVIDENCE_GROUNDED,
        route=route,
        citations=[],
        retrieved_chunks=chunks or [],
        agent_outputs={},
        metrics={},
        trace=[],
    )


# --- summarize: union + NaN-skip + denominator ---


def test_summarize_union_skips_nans_and_exposes_n():
    records = [
        {"a": 1.0, "b": math.nan},
        {"a": 3.0, "b": 4.0, "c": 2.0},
        {"a": math.nan, "b": 6.0},
    ]
    summary = summarize(records)
    # a: nan skipped -> mean(1, 3) = 2; n=2
    assert summary["a"] == pytest.approx(2.0)
    assert summary["a_n"] == 2.0
    # b: nan skipped -> mean(4, 6) = 5; n=2
    assert summary["b"] == pytest.approx(5.0)
    assert summary["b_n"] == 2.0
    # c only present in one record; union brings it in
    assert summary["c"] == pytest.approx(2.0)
    assert summary["c_n"] == 1.0


def test_summarize_all_nan_emits_nan_with_zero_n():
    summary = summarize([{"x": math.nan}, {"x": math.nan}])
    assert math.isnan(summary["x"])
    assert summary["x_n"] == 0.0


# --- rubric_coverage: rare-token gate suppresses verbosity wins ---


def test_rubric_coverage_requires_rare_token_hit():
    rubric = ["The student should reference photosynthesis and chloroplasts explicitly."]
    # Answer that repeats every stopword but never mentions rare tokens: should NOT count as covered.
    verbose_no_rare = (
        "the student should the student should the student should and and and should should and student"
    )
    assert rubric_coverage(verbose_no_rare, rubric) == 0.0
    # Answer that references a rare token counts.
    with_rare = "The student should mention chloroplasts and photosynthesis."
    assert rubric_coverage(with_rare, rubric) == 1.0


def test_rubric_coverage_verbatim_prefix_matches():
    rubric = ["Explain why the sky appears blue in clear daylight."]
    answer = "Explain why the sky appears blue in clear daylight due to Rayleigh scattering."
    assert rubric_coverage(answer, rubric) == 1.0


# --- tutoreval_keypoint_hit_rate: soft curve ---


def test_hit_rate_soft_curve_interpolates():
    # Key point has 4 tokens; answer has 2 of them -> overlap 0.5 -> on the 0.25..0.7 ramp.
    key = ["alpha beta gamma delta"]
    ans = "alpha beta"
    score = tutoreval_keypoint_hit_rate(ans, key)
    expected = (0.5 - 0.25) / (0.7 - 0.25)
    assert score == pytest.approx(expected, rel=1e-3)


def test_hit_rate_zero_below_floor():
    key = ["alpha beta gamma delta epsilon zeta eta theta"]
    ans = "alpha"  # 1/8 = 0.125 < 0.25
    assert tutoreval_keypoint_hit_rate(ans, key) == 0.0


def test_hit_rate_full_credit_above_threshold():
    key = ["alpha beta gamma delta"]
    ans = "alpha beta gamma"  # 3/4 = 0.75 >= 0.7
    assert tutoreval_keypoint_hit_rate(ans, key) == 1.0


# --- keypoint_token_alignment: micro-averaged precision ---


def test_keypoint_precision_micro_averaged():
    # Two keypoints. Answer hits all of kp1 tokens and some of kp2.
    key_points = ["apples oranges", "bananas grapes"]
    answer = "apples oranges bananas"
    precision, recall = keypoint_token_alignment(answer, key_points)
    # per-kp precision: kp1 -> hits={apples, oranges}, extras={bananas} -> 2/3
    #                  kp2 -> hits={bananas}, extras={apples, oranges} -> 1/3
    # mean -> 0.5
    assert precision == pytest.approx(0.5, rel=1e-3)
    # recall: answer ∩ union = {apples, oranges, bananas}; union = 4 tokens -> 3/4
    assert recall == pytest.approx(0.75)


# --- edu_score_alignment: NaN for ungradeable ---


def test_edu_score_alignment_nan_when_no_reference():
    val = edu_score_alignment('{"score": 80}', None)
    assert math.isnan(val)


def test_edu_score_alignment_scales_with_delta():
    val = edu_score_alignment('{"score": 80}', 90.0)
    assert val == pytest.approx(0.9, rel=1e-3)


# --- tutoreval_chapter_grounding: NaN for non-chapter ---


def test_chapter_grounding_nan_when_not_applicable():
    ex = _example(metadata={})
    resp = _response("anything")
    assert math.isnan(tutoreval_chapter_grounding(ex, resp, "anything"))


def test_chapter_grounding_applies_when_in_chapter():
    ex = _example(
        metadata={"tutoreval_answer_in_chapter": True},
        context_text="photosynthesis converts light into chemical energy",
    )
    resp = _response("photosynthesis energy")
    val = tutoreval_chapter_grounding(ex, resp, "photosynthesis energy")
    assert 0.0 < val <= 1.0


# --- compute_metrics + summarize end-to-end for edubench_consensus ---


def test_compute_metrics_emits_12d_keys_for_edubench_consensus():
    ex = _example(
        metadata={
            "evaluation_profile": "edubench_consensus",
            "edubench_reference_score_mean": 80.0,
            "information": {"Subject": "Biology", "Task": "Evaluate student answer"},
        },
        rubric=["The student should mention chloroplasts."],
    )
    ans = '{"score": 78, "Scoring_Details": {"accuracy": "good"}, "Personalized Feedback": "Mention chloroplasts explicitly."}'
    resp = _response(ans)
    metrics = compute_metrics(ex, resp)
    for key in [
        "edubench_iftc",
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
        "edubench_12d_mean",
        "edubench_scenario_adaptation",
        "edubench_factual_reasoning_accuracy",
        "edubench_pedagogical_application",
    ]:
        assert key in metrics, f"missing {key}"
    assert 0.0 <= metrics["edubench_12d_mean"] <= 1.0
    assert metrics["edu_json_compliance"] == pytest.approx(1.0)


def test_summarize_skips_ungradeable_chapter_rows():
    ex_in = _example(
        metadata={
            "evaluation_profile": "tutoreval_key_points",
            "tutoreval_answer_in_chapter": True,
            "tutoreval_key_points": ["photosynthesis energy"],
        },
        context_text="photosynthesis energy conversion",
    )
    ex_out = _example(
        metadata={
            "evaluation_profile": "tutoreval_key_points",
            "tutoreval_key_points": ["photosynthesis energy"],
        },
    )
    r1 = compute_metrics(ex_in, _response("photosynthesis energy"))
    r2 = compute_metrics(ex_out, _response("photosynthesis energy"))
    summary = summarize([r1, r2])
    # Non-chapter row contributes NaN -> only the in-chapter row counts.
    assert summary["tutoreval_chapter_grounding_n"] == 1.0

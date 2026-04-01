from __future__ import annotations

from eduagentic.config import DEFAULT_CONFIG
from eduagentic.core.contracts import BenchmarkExample, TaskRegime
from eduagentic.ml.regime_router import LightweightRegimeRouter


def test_router_prefers_evidence_for_citation_task():
    router = LightweightRegimeRouter(DEFAULT_CONFIG)
    example = BenchmarkExample(
        example_id="1",
        dataset_name="HotpotQA",
        regime_hint=None,
        question="Explain the answer with citations and supporting facts from the source documents.",
    )
    decision = router.decide(example)
    assert decision.require_retrieval is True
    assert decision.regime == TaskRegime.EVIDENCE_GROUNDED


def test_router_prefers_non_rag_for_rubric_feedback():
    router = LightweightRegimeRouter(DEFAULT_CONFIG)
    example = BenchmarkExample(
        example_id="2",
        dataset_name="TutorBench",
        regime_hint=None,
        question="Give rubric-based feedback to the student and identify the misconception.",
        rubric=["Identify misconception", "Give actionable next step"],
    )
    decision = router.decide(example)
    assert decision.use_rubric_agent is True
    assert decision.regime == TaskRegime.RUBRIC_FEEDBACK

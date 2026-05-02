"""Smoke tests for the ablation flags, corpus_factuality metric, and stats script."""
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from eduagentic.agents.base import AgentDependencies
from eduagentic.config import DEFAULT_CONFIG, PipelineConfig, RouterConfig
from eduagentic.core.contracts import (
    ArchitectureFamily,
    BenchmarkExample,
    ModelResponse,
    RouteDecision,
    TaskRegime,
)
from eduagentic.evaluation.metrics import compute_metrics, corpus_factuality, summarize
from eduagentic.orchestration.pipelines import (
    HybridFastPipeline,
    MultiAgentNoRAGPipeline,
    SingleAgentNoRAGPipeline,
)
from eduagentic.retrieval.corpus import SourceDocument, chunk_documents
from eduagentic.retrieval.index import HybridIndex
from eduagentic.retrieval.packer import ContextPacker
from eduagentic.retrieval.reranker import LightweightReranker
from eduagentic.tools import ContextToolExecutor


class _FakeChatClient:
    async def chat(self, **kwargs):
        messages = kwargs["messages"]
        prompt = messages[-1].content if hasattr(messages[-1], "content") else messages[-1]["content"]
        return ModelResponse(
            text=f"FAKE_ANSWER\n{str(prompt)[:120]}",
            model=kwargs["model"],
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            raw={"ok": True},
            latency_ms=7,
        )


def _make_deps(index: HybridIndex | None = None) -> AgentDependencies:
    deps = AgentDependencies(
        text_client=_FakeChatClient(),
        vision_client=_FakeChatClient(),
        text_model="tiny-text",
        vision_model="tiny-vision",
        retriever=index,
        reranker=LightweightReranker(),
        packer=ContextPacker(max_chars=800),
    )
    deps.tools = ContextToolExecutor(deps)
    return deps


def _make_index() -> HybridIndex:
    docs = [
        SourceDocument(doc_id="d1", title="Photosynthesis", text="Photosynthesis converts light energy into chemical energy in plants."),
        SourceDocument(doc_id="d2", title="Radiation", text="Radiation is the fastest mode of heat transfer and travels through empty space."),
    ]
    return HybridIndex().fit(chunk_documents(docs, chunk_size=20, chunk_overlap=5))


def _make_route(require_retrieval: bool = False, use_critic: bool = True) -> RouteDecision:
    return RouteDecision(
        regime=TaskRegime.ADAPTIVE_TUTORING,
        architecture=ArchitectureFamily.HYBRID_FAST,
        require_retrieval=require_retrieval,
        use_critic=use_critic,
        use_rubric_agent=False,
        specialist_roles=["planner", "tutor"],
        scores={"evidence": 0.2},
    )


def _make_example() -> BenchmarkExample:
    return BenchmarkExample(
        example_id="x",
        dataset_name="EduBench",
        regime_hint=None,
        question="Explain photosynthesis with evidence.",
    )


def _config_with(pipeline_overrides: dict | None = None, router_overrides: dict | None = None):
    pipeline = replace(DEFAULT_CONFIG.pipeline, **(pipeline_overrides or {}))
    router = replace(DEFAULT_CONFIG.router, **(router_overrides or {}))
    return replace(DEFAULT_CONFIG, pipeline=pipeline, router=router)


@pytest.mark.asyncio
async def test_hybrid_force_retrieval_invokes_retriever_when_route_says_no() -> None:
    index = _make_index()
    deps = _make_deps(index=index)
    config = _config_with({"hybrid_force_retrieval": True})
    pipeline = HybridFastPipeline(config, deps)
    route = _make_route(require_retrieval=False)
    response = await pipeline.run(_make_example(), route)
    # Forced retrieval should have produced chunks even though the route says no.
    assert response.retrieved_chunks, "force_retrieval must trigger retriever"
    assert response.metrics.get("ablation.hybrid_force_retrieval") == 1.0


@pytest.mark.asyncio
async def test_hybrid_disable_critic_removes_critic_agent() -> None:
    index = _make_index()
    deps = _make_deps(index=index)
    config = _config_with({"hybrid_disable_critic": True})
    pipeline = HybridFastPipeline(config, deps)
    route = _make_route(require_retrieval=True, use_critic=True)
    response = await pipeline.run(_make_example(), route)
    assert "critic" not in response.agent_outputs
    assert response.metrics.get("ablation.hybrid_disable_critic") == 1.0


@pytest.mark.asyncio
async def test_non_rag_enable_retrieval_runs_retriever() -> None:
    index = _make_index()
    deps = _make_deps(index=index)
    config = _config_with({"non_rag_enable_retrieval": True})
    pipeline = MultiAgentNoRAGPipeline(config, deps)
    route = RouteDecision(
        regime=TaskRegime.ADAPTIVE_TUTORING,
        architecture=ArchitectureFamily.NON_RAG_MULTI_AGENT,
        require_retrieval=False,
        use_critic=False,
        use_rubric_agent=False,
    )
    response = await pipeline.run(_make_example(), route)
    assert response.retrieved_chunks, "non_rag_enable_retrieval must pull chunks"
    assert response.metrics.get("ablation.non_rag_enable_retrieval") == 1.0


@pytest.mark.asyncio
async def test_disable_critic_global_overrides_every_pipeline() -> None:
    index = _make_index()
    deps = _make_deps(index=index)
    config = _config_with({"disable_critic_global": True})
    pipeline = SingleAgentNoRAGPipeline(config, deps)
    route = RouteDecision(
        regime=TaskRegime.ADAPTIVE_TUTORING,
        architecture=ArchitectureFamily.SINGLE_AGENT_NO_RAG,
        require_retrieval=False,
        use_critic=True,
        use_rubric_agent=False,
    )
    response = await pipeline.run(_make_example(), route)
    assert "critic" not in response.agent_outputs
    assert response.metrics.get("ablation.disable_critic_global") == 1.0


@pytest.mark.asyncio
async def test_default_hybrid_config_preserves_production_behavior() -> None:
    """Sanity guard: all ablation flags default False => same agents as before."""
    index = _make_index()
    deps = _make_deps(index=index)
    pipeline = HybridFastPipeline(DEFAULT_CONFIG, deps)
    route = _make_route(require_retrieval=True, use_critic=True)
    response = await pipeline.run(_make_example(), route)
    for flag in (
        "ablation.hybrid_force_retrieval",
        "ablation.hybrid_disable_critic",
        "ablation.non_rag_enable_retrieval",
        "ablation.disable_critic_global",
    ):
        assert response.metrics.get(flag) == 0.0, f"{flag} must default off"
    # Critic must still run because the route says so and no ablation disabled it.
    assert "critic" in response.agent_outputs


def test_corpus_factuality_returns_nan_without_index() -> None:
    assert math.isnan(corpus_factuality("Photosynthesis converts light energy into chemical energy.", None))


def test_corpus_factuality_scores_sentences_against_shared_index() -> None:
    index = _make_index()
    answer = (
        "Photosynthesis converts light energy into chemical energy. "
        "Radiation is the fastest mode of heat transfer."
    )
    score = corpus_factuality(answer, index)
    assert not math.isnan(score), "score must be populated when sentences match the corpus"
    assert 0.0 <= score <= 1.0


def test_compute_metrics_populates_corpus_factuality_when_index_supplied() -> None:
    from eduagentic.core.contracts import PipelineResponse

    example = BenchmarkExample(
        example_id="cf",
        dataset_name="TutorEval",
        regime_hint=TaskRegime.ADAPTIVE_TUTORING,
        question="What is the fastest mode of heat transfer?",
        gold_answer="Radiation is the fastest mode of heat transfer.",
    )
    response = PipelineResponse(
        answer="Radiation is the fastest mode of heat transfer in empty space.",
        architecture=ArchitectureFamily.HYBRID_FAST,
        regime=TaskRegime.ADAPTIVE_TUTORING,
        route=_make_route(),
        metrics={"latency_ms": 100.0, "agent_count": 2.0},
    )
    metrics = compute_metrics(example, response, corpus_index=_make_index())
    assert not math.isnan(metrics["corpus_factuality"])
    # summarize must not crash on the new key even with mixed NaN rows.
    summary = summarize([metrics, {"corpus_factuality": float("nan")}])
    assert "corpus_factuality" in summary


def test_compute_paired_stats_against_tiny_matched_slice(tmp_path: Path) -> None:
    """End-to-end check that the stats script handles aligned per-example records."""
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "compute_paired_stats.py"
    spec = importlib.util.spec_from_file_location("compute_paired_stats_test", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    session_a = {
        "result": {
            "records": [
                {"example_id": "e1", "metrics": {"token_f1": 0.10, "rubric_coverage": 0.8}},
                {"example_id": "e2", "metrics": {"token_f1": 0.20, "rubric_coverage": 0.9}},
                {"example_id": "e3", "metrics": {"token_f1": 0.15, "rubric_coverage": 0.85}},
            ]
        }
    }
    session_b = {
        "result": {
            "records": [
                {"example_id": "e1", "metrics": {"token_f1": 0.13, "rubric_coverage": 0.82}},
                {"example_id": "e2", "metrics": {"token_f1": 0.22, "rubric_coverage": 0.91}},
                {"example_id": "e3", "metrics": {"token_f1": 0.17, "rubric_coverage": 0.86}},
            ]
        }
    }
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    path_a.write_text(json.dumps(session_a))
    path_b.write_text(json.dumps(session_b))

    records_a = module._records_by_id(module._load_records(path_a))  # noqa: SLF001
    records_b = module._records_by_id(module._load_records(path_b))  # noqa: SLF001
    rows = module.compute_paired_stats(
        records_a,
        records_b,
        metric_keys=["token_f1", "rubric_coverage"],
        bootstrap=200,
        permutations=200,
        confidence=0.90,
        seed=7,
    )
    assert {row["metric"] for row in rows} == {"token_f1", "rubric_coverage"}
    for row in rows:
        assert row["n_paired"] == 3
        assert row["mean_delta"] > 0.0  # session B is strictly higher on every paired example.
        assert row["ci_low"] <= row["mean_delta"] <= row["ci_high"]

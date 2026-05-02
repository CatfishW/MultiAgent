from __future__ import annotations

import pytest

from eduagentic.agents.base import AgentDependencies
from eduagentic.config import DEFAULT_CONFIG
from eduagentic.core.contracts import BenchmarkExample, ModelResponse, RouteDecision, TaskRegime, ArchitectureFamily
from eduagentic.orchestration.pipelines import HybridFastPipeline
from eduagentic.retrieval.corpus import SourceDocument, chunk_documents
from eduagentic.retrieval.index import HybridIndex
from eduagentic.retrieval.packer import ContextPacker
from eduagentic.retrieval.reranker import LightweightReranker
from eduagentic.tools import ContextToolExecutor


class FakeChatClient:
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


@pytest.mark.asyncio
async def test_hybrid_pipeline_smoke():
    docs = [SourceDocument(doc_id="d1", title="Reference", text="Photosynthesis converts light energy into chemical energy in plants.")]
    index = HybridIndex().fit(chunk_documents(docs, chunk_size=20, chunk_overlap=5))
    deps = AgentDependencies(
        text_client=FakeChatClient(),
        vision_client=FakeChatClient(),
        text_model="tiny-text",
        vision_model="tiny-vision",
        retriever=index,
        reranker=LightweightReranker(),
        packer=ContextPacker(max_chars=800),
    )
    deps.tools = ContextToolExecutor(deps)
    pipeline = HybridFastPipeline(DEFAULT_CONFIG, deps)
    example = BenchmarkExample(
        example_id="1",
        dataset_name="EduBench",
        regime_hint=None,
        question="Explain photosynthesis with evidence.",
    )
    route = RouteDecision(
        regime=TaskRegime.EVIDENCE_GROUNDED,
        architecture=ArchitectureFamily.HYBRID_FAST,
        require_retrieval=True,
        use_critic=True,
        use_rubric_agent=False,
        specialist_roles=["planner", "retriever", "tutor", "critic"],
    )
    response = await pipeline.run(example, route)
    assert "FAKE_ANSWER" in response.answer
    assert response.retrieved_chunks
    assert response.metrics["agent_count"] >= 3
    assert response.metrics["api_time_ms"] > 0
    assert response.metrics["llm_call_count"] >= 1
    assert response.metrics["total_tokens"] >= 15
    assert response.metrics["complexity_units"] > 0
    assert response.metrics["tool_call_count"] >= 1
    assert response.agent_outputs["retriever"].artifacts["mode"] == "tool_search"

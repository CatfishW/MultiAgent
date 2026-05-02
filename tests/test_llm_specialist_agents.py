from __future__ import annotations

import json

import pytest

from eduagentic.agents.base import AgentContext, AgentDependencies
from eduagentic.agents.critic import CriticAgent
from eduagentic.agents.diagnoser import DiagnoserAgent
from eduagentic.agents.planner import PlannerAgent
from eduagentic.agents.retriever import RetrieverAgent
from eduagentic.agents.rubric import RubricAgent
from eduagentic.agents.tutor import TutorAgent
from eduagentic.config import DEFAULT_CONFIG
from eduagentic.core.contracts import ArchitectureFamily, BenchmarkExample, ModelResponse, RouteDecision, TaskRegime
from eduagentic.retrieval.corpus import SourceDocument, chunk_documents
from eduagentic.retrieval.index import HybridIndex
from eduagentic.retrieval.packer import ContextPacker
from eduagentic.retrieval.reranker import LightweightReranker
from eduagentic.tools import ContextToolExecutor


class RecordingChatClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        system = kwargs["messages"][0].content
        if "planning agent" in system:
            text = json.dumps({"strategy": "1. inspect\n2. answer", "queries": ["photosynthesis evidence"]})
        elif "state diagnosis agent" in system:
            text = json.dumps(
                {
                    "level": "beginner",
                    "goals": ["understand current task"],
                    "misconceptions": ["thinks plants eat sunlight"],
                    "preferred_style": "step-by-step",
                    "summary": "beginner needs step-by-step support",
                }
            )
        elif "criteria analysis agent" in system:
            text = json.dumps({"summary": "Use evidence and be clear.", "criteria": ["correctness", "evidence"]})
        elif "retrieval query agent" in system:
            text = json.dumps({"queries": ["photosynthesis chemical energy"], "rationale": "Need source evidence."})
        else:
            text = "LLM_FINAL"
        return ModelResponse(
            text=text,
            model=kwargs["model"],
            usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            raw={"ok": True},
            latency_ms=5,
        )


def _route() -> RouteDecision:
    return RouteDecision(
        regime=TaskRegime.EVIDENCE_GROUNDED,
        architecture=ArchitectureFamily.HYBRID_FAST,
        require_retrieval=True,
        use_critic=True,
        use_rubric_agent=True,
        specialist_roles=["planner", "diagnoser", "rubric", "retriever", "tutor", "critic"],
        scores={"evidence": 0.8},
    )


def _context(deps: AgentDependencies) -> AgentContext:
    example = BenchmarkExample(
        example_id="llm-agent",
        dataset_name="custom-domain",
        regime_hint=None,
        question="Explain photosynthesis with evidence.",
        rubric=["Use evidence", "Be clear"],
    )
    return AgentContext(example=example, route=_route(), budget=DEFAULT_CONFIG.budget)


@pytest.mark.asyncio
async def test_specialist_agents_are_llm_backed_when_client_configured() -> None:
    client = RecordingChatClient()
    index = HybridIndex().fit(
        chunk_documents(
            [
                SourceDocument(
                    doc_id="d1",
                    title="Photosynthesis",
                    text="Photosynthesis converts light energy into chemical energy in plants.",
                )
            ],
            chunk_size=20,
            chunk_overlap=5,
        )
    )
    deps = AgentDependencies(
        text_client=client,
        text_model="tiny-text",
        retriever=index,
        reranker=LightweightReranker(),
        packer=ContextPacker(max_chars=800),
    )
    deps.tools = ContextToolExecutor(deps)
    context = _context(deps)

    planner = await PlannerAgent(deps).run(context)
    context.plan_text = planner.text
    context.search_queries = planner.artifacts["queries"]
    diagnoser = await DiagnoserAgent(deps).run(context)
    context.student_state = diagnoser.artifacts["student_state"]
    rubric = await RubricAgent(deps).run(context)
    context.rubric_summary = rubric.text
    retriever = await RetrieverAgent(deps).run(context)
    context.retrieved_chunks = retriever.artifacts["chunks"]
    tutor = await TutorAgent(deps).run(context)
    context.draft_answer = tutor.text
    critic = await CriticAgent(deps).run(context)

    assert planner.artifacts["mode"] == "llm"
    assert diagnoser.artifacts["mode"] == "llm"
    assert rubric.artifacts["mode"] == "llm"
    assert planner.artifacts["tool_observations"]
    assert diagnoser.artifacts["tool_observations"]
    assert rubric.artifacts["tool_observations"]
    assert retriever.artifacts["mode"] == "llm"
    assert tutor.text == "LLM_FINAL"
    assert critic.text == "LLM_FINAL"
    assert len(client.calls) == 6

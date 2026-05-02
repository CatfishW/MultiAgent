from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..core.contracts import AgentResult, BenchmarkExample, BudgetPolicy, RetrievedChunk, RouteDecision, StudentState
from ..retrieval.index import HybridIndex
from ..retrieval.packer import ContextPacker
from ..retrieval.reranker import LightweightReranker


class ChatClient(Protocol):
    async def chat(self, **kwargs: Any) -> Any: ...


@dataclass(slots=True)
class AgentDependencies:
    text_client: ChatClient | None = None
    vision_client: ChatClient | None = None
    text_model: str | None = None
    vision_model: str | None = None
    text_chat_extra: dict[str, Any] = field(default_factory=dict)
    vision_chat_extra: dict[str, Any] = field(default_factory=dict)
    retriever: HybridIndex | None = None
    reranker: LightweightReranker | None = None
    packer: ContextPacker | None = None
    tools: Any | None = None


@dataclass(slots=True)
class AgentContext:
    example: BenchmarkExample
    route: RouteDecision
    budget: BudgetPolicy
    student_state: StudentState | None = None
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    plan_text: str | None = None
    search_queries: list[str] = field(default_factory=list)
    rubric_summary: str | None = None
    draft_answer: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)


class BaseAgent:
    role_name = "base"

    def __init__(self, deps: AgentDependencies) -> None:
        self.deps = deps

    async def run(self, context: AgentContext) -> AgentResult:
        raise NotImplementedError

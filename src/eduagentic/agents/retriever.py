from __future__ import annotations

from ..core.contracts import AgentResult
from ..retrieval.packer import ContextPacker
from .base import AgentContext, BaseAgent


class RetrieverAgent(BaseAgent):
    role_name = "retriever"

    async def run(self, context: AgentContext) -> AgentResult:
        if self.deps.retriever is None:
            return AgentResult(role=self.role_name, text="retrieval disabled", confidence=0.0)
        queries = context.search_queries or [context.example.question]
        merged = []
        for query in queries[: context.budget.max_retrieval_queries]:
            result = self.deps.retriever.search(query)
            merged.extend(result.chunks)
        dedup: dict[str, object] = {}
        for chunk in merged:
            previous = dedup.get(chunk.chunk_id)
            if previous is None or chunk.score > previous.score:
                dedup[chunk.chunk_id] = chunk
        ranked = list(dedup.values())
        if self.deps.reranker is not None:
            ranked = self.deps.reranker.rerank(context.example.question, ranked)
        packer = self.deps.packer or ContextPacker(max_chars=context.budget.max_context_chars)
        selected = packer.select(context.example.question, ranked, final_k=min(context.budget.max_tool_calls, 6))
        rendered = packer.render_context(selected)
        citations = [item.doc_id for item in selected]
        return AgentResult(
            role=self.role_name,
            text=rendered,
            confidence=0.84 if selected else 0.25,
            citations=citations,
            artifacts={"chunks": selected, "queries": queries},
        )

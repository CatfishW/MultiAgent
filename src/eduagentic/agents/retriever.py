from __future__ import annotations

import json
import re
import time
from typing import Any

from ..core.contracts import AgentResult, ModelMessage
from ..prompts.templates import RETRIEVER_SYSTEM_PROMPT
from ..retrieval.packer import ContextPacker
from .base import AgentContext, BaseAgent


class RetrieverAgent(BaseAgent):
    role_name = "retriever"

    def _render_prompt(self, context: AgentContext) -> str:
        example = context.example
        parts = [
            f"Domain or dataset: {example.dataset_name}",
            f"Question or task:\n{example.question}",
        ]
        if context.plan_text:
            parts.append(f"Planner strategy:\n{context.plan_text}")
        if context.search_queries:
            parts.append("Initial planner queries:\n" + "\n".join(f"- {query}" for query in context.search_queries[: context.budget.max_retrieval_queries]))
        if context.rubric_summary:
            parts.append(f"Criteria summary:\n{context.rubric_summary}")
        if context.student_state is not None:
            parts.append(f"Visible state summary:\n{context.student_state.summary}")
        parts.append(f"Return at most {context.budget.max_retrieval_queries} search queries.")
        return "\n\n".join(parts)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"\s*```$", "", stripped)
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
            if not match:
                return None
            try:
                value = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return value if isinstance(value, dict) else None

    async def _llm_queries(self, context: AgentContext, fallback: list[str]) -> tuple[list[str], dict[str, Any], int | None]:
        if self.deps.text_client is None or self.deps.text_model is None:
            return fallback, {"mode": "heuristic_fallback"}, None
        response = await self.deps.text_client.chat(
            model=self.deps.text_model,
            messages=[
                ModelMessage(role="system", content=RETRIEVER_SYSTEM_PROMPT),
                ModelMessage(role="user", content=self._render_prompt(context)),
            ],
            temperature=0.0,
            max_tokens=220,
            extra=self.deps.text_chat_extra or None,
        )
        payload = self._extract_json(response.text)
        raw_queries = payload.get("queries") if isinstance(payload, dict) else None
        query_items = raw_queries if isinstance(raw_queries, list) else []
        queries: list[str] = []
        seen: set[str] = set()
        for item in [*query_items, *fallback]:
            query = str(item).strip()
            if not query or query in seen:
                continue
            queries.append(query)
            seen.add(query)
            if len(queries) >= context.budget.max_retrieval_queries:
                break
        artifacts = {
            "mode": "llm" if isinstance(payload, dict) else "llm_parse_fallback",
            "usage": response.usage,
            "raw": response.raw,
            "query_rationale": payload.get("rationale") if isinstance(payload, dict) else None,
        }
        return queries or fallback, artifacts, response.latency_ms

    async def run(self, context: AgentContext) -> AgentResult:
        started = time.perf_counter()
        if self.deps.retriever is None:
            return AgentResult(
                role=self.role_name,
                text="retrieval disabled",
                confidence=0.0,
                artifacts={"chunks": [], "queries": [], "query_count": 0},
                latency_ms=0,
            )
        fallback_queries = context.search_queries or [context.example.question]
        queries, llm_artifacts, llm_latency_ms = await self._llm_queries(context, fallback_queries)
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
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        artifacts = {
            "chunks": selected,
            "queries": queries,
            "query_count": len(queries),
            **llm_artifacts,
        }
        if llm_latency_ms is not None:
            artifacts["llm_query_latency_ms"] = llm_latency_ms
        return AgentResult(
            role=self.role_name,
            text=rendered,
            confidence=0.84 if selected else 0.25,
            citations=citations,
            artifacts=artifacts,
            latency_ms=elapsed_ms,
        )

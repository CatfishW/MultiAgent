from __future__ import annotations

from dataclasses import dataclass, field
import re
import time
from typing import Any, Iterable

from ..agents.base import AgentContext, AgentDependencies
from ..core.contracts import RetrievedChunk
from ..retrieval.packer import ContextPacker
from ..utils.text import sentence_overlap, tokenize


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""


@dataclass(slots=True)
class ToolObservation:
    name: str
    content: str
    call_id: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0


def _clean_text(value: Any, *, max_chars: int = 1200) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:max_chars]


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part.strip()]


def normalize_tool_calls(value: Any, *, allowed: set[str] | None = None, max_calls: int = 4) -> list[ToolCall]:
    if not isinstance(value, list):
        return []
    calls: list[ToolCall] = []
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("tool") or "").strip()
        if not name or (allowed is not None and name not in allowed):
            continue
        args = item.get("arguments", item.get("args", {}))
        if not isinstance(args, dict):
            args = {}
        call_id = str(item.get("id") or item.get("call_id") or f"call_{idx + 1}")
        calls.append(ToolCall(name=name, arguments=dict(args), call_id=call_id))
        if len(calls) >= max_calls:
            break
    return calls


def render_tool_observations(observations: Iterable[ToolObservation], *, max_chars: int = 1400) -> str:
    parts: list[str] = []
    remaining = max_chars
    for obs in observations:
        if remaining <= 0:
            break
        line = f"- {obs.name}: {_clean_text(obs.content, max_chars=remaining)}"
        parts.append(line[:remaining])
        remaining -= len(parts[-1])
    return "\n".join(parts)


class ContextToolExecutor:
    """Bounded, side-effect-free tools available to MARLET agents.

    The tools are local Python functions, not open-ended shell or network access.
    They expose small observations that can be placed into the final prompt.
    """

    TOOL_NAMES = {
        "extract_key_terms",
        "inspect_inline_context",
        "inspect_dialogue_state",
        "list_answer_criteria",
        "search_corpus",
    }

    def __init__(self, deps: AgentDependencies) -> None:
        self.deps = deps

    def default_calls_for_role(self, context: AgentContext, role: str) -> list[ToolCall]:
        if role == "planner":
            calls = [
                ToolCall(
                    name="extract_key_terms",
                    arguments={"text": context.example.question, "limit": 8},
                    call_id="planner_terms",
                )
            ]
            if context.example.context_text:
                calls.append(
                    ToolCall(
                        name="inspect_inline_context",
                        arguments={"query": context.example.question, "max_chars": 700},
                        call_id="planner_inline_context",
                    )
                )
            return calls
        if role == "diagnoser":
            return [ToolCall(name="inspect_dialogue_state", call_id="diagnoser_state")]
        if role == "rubric":
            return [ToolCall(name="list_answer_criteria", call_id="criteria_list")]
        return []

    def execute(self, context: AgentContext, calls: Iterable[ToolCall], *, max_calls: int | None = None) -> list[ToolObservation]:
        limit = max(0, max_calls if max_calls is not None else context.budget.max_tool_calls)
        observations: list[ToolObservation] = []
        for call in list(calls)[:limit]:
            started = time.perf_counter()
            try:
                obs = self._execute_one(context, call)
            except Exception as exc:  # Tools should never crash the pipeline.
                obs = ToolObservation(
                    name=call.name,
                    call_id=call.call_id,
                    content=f"tool failed: {type(exc).__name__}",
                    artifacts={"error": str(exc)},
                )
            obs.latency_ms = int((time.perf_counter() - started) * 1000)
            observations.append(obs)
        return observations

    def retrieve_with_queries(self, context: AgentContext, queries: list[str]) -> tuple[list[RetrievedChunk], str, list[ToolObservation]]:
        calls = [
            ToolCall(
                name="search_corpus",
                arguments={"query": query, "top_k": max(context.budget.max_tool_calls, 4)},
                call_id=f"search_{idx + 1}",
            )
            for idx, query in enumerate(queries[: context.budget.max_retrieval_queries])
            if str(query).strip()
        ]
        observations = self.execute(context, calls, max_calls=context.budget.max_retrieval_queries)
        merged: dict[str, RetrievedChunk] = {}
        for obs in observations:
            chunks = obs.artifacts.get("chunks", [])
            if not isinstance(chunks, list):
                continue
            for chunk in chunks:
                if not isinstance(chunk, RetrievedChunk):
                    continue
                previous = merged.get(chunk.chunk_id)
                if previous is None or chunk.score > previous.score:
                    merged[chunk.chunk_id] = chunk
        ranked = list(merged.values())
        if self.deps.reranker is not None:
            ranked = self.deps.reranker.rerank(context.example.question, ranked)
        packer = self.deps.packer or ContextPacker(max_chars=context.budget.max_context_chars)
        selected = packer.select(context.example.question, ranked, final_k=min(context.budget.max_tool_calls, 6))
        return selected, packer.render_context(selected), observations

    def _execute_one(self, context: AgentContext, call: ToolCall) -> ToolObservation:
        if call.name == "extract_key_terms":
            return self._extract_key_terms(context, call)
        if call.name == "inspect_inline_context":
            return self._inspect_inline_context(context, call)
        if call.name == "inspect_dialogue_state":
            return self._inspect_dialogue_state(context, call)
        if call.name == "list_answer_criteria":
            return self._list_answer_criteria(context, call)
        if call.name == "search_corpus":
            return self._search_corpus(context, call)
        return ToolObservation(name=call.name, call_id=call.call_id, content="unknown tool")

    def _extract_key_terms(self, context: AgentContext, call: ToolCall) -> ToolObservation:
        text = _clean_text(call.arguments.get("text") or context.example.question, max_chars=1800)
        limit = min(12, max(1, int(call.arguments.get("limit", 8) or 8)))
        seen: set[str] = set()
        terms: list[str] = []
        for token in tokenize(text):
            if len(token) < 4 or token in seen:
                continue
            terms.append(token)
            seen.add(token)
            if len(terms) >= limit:
                break
        return ToolObservation(
            name=call.name,
            call_id=call.call_id,
            content=", ".join(terms) if terms else "no salient terms found",
            artifacts={"terms": terms},
        )

    def _inspect_inline_context(self, context: AgentContext, call: ToolCall) -> ToolObservation:
        source = context.example.context_text or ""
        if not source.strip():
            return ToolObservation(name=call.name, call_id=call.call_id, content="no inline context provided")
        query = _clean_text(call.arguments.get("query") or context.example.question, max_chars=500)
        q_tokens = set(tokenize(query))
        scored: list[tuple[float, str]] = []
        for sent in _sentences(source):
            s_tokens = set(tokenize(sent))
            score = len(q_tokens & s_tokens) / max(1, len(q_tokens))
            if score > 0:
                scored.append((score, sent))
        if not scored:
            snippet = source[: int(call.arguments.get("max_chars", 700) or 700)]
        else:
            scored.sort(key=lambda item: item[0], reverse=True)
            snippet = " ".join(sent for _, sent in scored[:3])
            snippet = snippet[: int(call.arguments.get("max_chars", 700) or 700)]
        return ToolObservation(name=call.name, call_id=call.call_id, content=snippet)

    def _inspect_dialogue_state(self, context: AgentContext, call: ToolCall) -> ToolObservation:
        turns = context.example.dialogue_history[-6:]
        if not turns:
            return ToolObservation(name=call.name, call_id=call.call_id, content="no prior interaction; state must come from current request")
        text = " ".join(f"{turn.role}: {turn.text}" for turn in turns)
        clues: list[str] = []
        lowered = text.lower()
        if any(word in lowered for word in ("confused", "stuck", "don't understand", "wrong")):
            clues.append("visible confusion or failed attempt")
        if any(word in lowered for word in ("beginner", "new to", "simple", "explain step")):
            clues.append("beginner-facing explanation likely useful")
        if any(word in lowered for word in ("brief", "concise", "short")):
            clues.append("concise style requested")
        content = "; ".join(clues) if clues else _clean_text(text, max_chars=350)
        return ToolObservation(name=call.name, call_id=call.call_id, content=content)

    def _list_answer_criteria(self, context: AgentContext, call: ToolCall) -> ToolObservation:
        criteria = [str(item).strip() for item in (context.example.rubric or []) if str(item).strip()]
        if not criteria:
            criteria = ["correctness", "clarity", "use evidence when available", "actionable next step"]
        return ToolObservation(
            name=call.name,
            call_id=call.call_id,
            content="; ".join(criteria[:8]),
            artifacts={"criteria": criteria[:10]},
        )

    def _search_corpus(self, context: AgentContext, call: ToolCall) -> ToolObservation:
        query = _clean_text(call.arguments.get("query") or context.example.question, max_chars=500)
        if self.deps.retriever is None:
            return ToolObservation(name=call.name, call_id=call.call_id, content="retrieval index unavailable", artifacts={"chunks": []})
        top_k = min(12, max(1, int(call.arguments.get("top_k", context.budget.max_tool_calls) or context.budget.max_tool_calls)))
        result = self.deps.retriever.search(query, top_k=top_k)
        chunks = list(result.chunks)
        rendered = []
        for chunk in chunks[: min(top_k, 4)]:
            rendered.append(f"[{chunk.doc_id}] {chunk.title}: {_clean_text(chunk.text, max_chars=220)}")
        content = "\n".join(rendered) if rendered else "no matching chunks"
        # Add a tiny diversity signal so downstream packing can avoid repeated observations.
        if len(chunks) >= 2:
            content += f"\nredundancy_hint={sentence_overlap(chunks[0].text, chunks[1].text):.3f}"
        return ToolObservation(name=call.name, call_id=call.call_id, content=content, artifacts={"chunks": chunks, "query": query})

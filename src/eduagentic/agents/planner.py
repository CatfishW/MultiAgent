from __future__ import annotations

import json
import re
from typing import Any

from ..core.contracts import AgentResult, ModelMessage, TaskRegime
from ..prompts.templates import PLANNER_SYSTEM_PROMPT
from ..tools import render_tool_observations
from ..tools.context_tools import normalize_tool_calls
from ..utils.text import tokenize
from .base import AgentContext, BaseAgent


class PlannerAgent(BaseAgent):
    role_name = "planner"

    def _heuristic_plan(self, context: AgentContext) -> tuple[str, list[str]]:
        example = context.example
        regime = context.route.regime
        queries = [example.question]
        key_terms = [token for token in tokenize(example.question) if len(token) > 4][:6]
        if key_terms:
            queries.append(" ".join(key_terms))
        if example.context_text:
            queries.append(f"{example.question} {example.dataset_name}")
        if example.rubric:
            queries.append(f"{example.question} {' '.join(example.rubric[:2])}")
        unique_queries = []
        seen = set()
        for query in queries:
            if query and query not in seen:
                unique_queries.append(query)
                seen.add(query)
        if regime == TaskRegime.EVIDENCE_GROUNDED:
            steps = [
                "Extract the core factual or explanatory target.",
                "Retrieve the smallest set of supporting passages.",
                "Answer directly and attach evidence markers.",
            ]
        elif regime == TaskRegime.RUBRIC_FEEDBACK:
            steps = [
                "Identify the task gap and the criteria dimensions.",
                "Explain what is correct, what is missing, and why.",
                "Give one or two high-value next steps, not a wall of feedback.",
            ]
        elif regime == TaskRegime.ADAPTIVE_TUTORING:
            steps = [
                "Infer visible user/task state and likely source of confusion.",
                "Choose the minimum explanation needed for progress.",
                "End with a check-for-understanding or next-step hint.",
            ]
        else:
            steps = [
                "Define the task goal and sequencing constraints.",
                "Sequence actions from diagnosis to guided practice to check.",
                "Keep the plan lightweight and actionable.",
            ]
        if context.route.require_retrieval:
            steps.insert(1, "Use retrieval only for the parts that need grounding.")
        return "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(steps)), unique_queries[: context.budget.max_retrieval_queries]

    def _render_prompt(self, context: AgentContext, tool_observations: list[Any] | None = None) -> str:
        example = context.example
        parts = [
            f"Domain or dataset: {example.dataset_name}",
            f"Response regime: {context.route.regime.value}",
            f"Retrieval gate: {'open' if context.route.require_retrieval else 'closed'}",
            f"Question or task:\n{example.question}",
        ]
        if example.context_text:
            parts.append(f"Inline context summary/source text:\n{example.context_text[:1800]}")
        if example.rubric:
            parts.append("Criteria:\n" + "\n".join(f"- {item}" for item in example.rubric[:8]))
        if example.dialogue_history:
            history = "\n".join(f"{turn.role}: {turn.text}" for turn in example.dialogue_history[-6:])
            parts.append(f"Recent interaction:\n{history}")
        if tool_observations:
            parts.append("Available tool observations:\n" + render_tool_observations(tool_observations, max_chars=900))
        parts.append(
            "Create a compact response strategy and retrieval queries. "
            f"Return at most {context.budget.max_retrieval_queries} queries."
        )
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

    def _normalize_llm_plan(self, payload: dict[str, Any], context: AgentContext) -> tuple[str, list[str]]:
        fallback_plan, fallback_queries = self._heuristic_plan(context)
        strategy = payload.get("strategy")
        if isinstance(strategy, list):
            strategy_text = "\n".join(f"{idx + 1}. {str(item).strip()}" for idx, item in enumerate(strategy) if str(item).strip())
        elif isinstance(strategy, str) and strategy.strip():
            strategy_text = strategy.strip()
        else:
            strategy_text = fallback_plan

        raw_queries = payload.get("queries")
        query_items = raw_queries if isinstance(raw_queries, list) else []
        queries: list[str] = []
        seen: set[str] = set()
        for item in [*query_items, *fallback_queries]:
            query = str(item).strip()
            if not query or query in seen:
                continue
            queries.append(query)
            seen.add(query)
            if len(queries) >= context.budget.max_retrieval_queries:
                break
        return strategy_text, queries or fallback_queries

    async def run(self, context: AgentContext) -> AgentResult:
        fallback_plan, fallback_queries = self._heuristic_plan(context)
        if self.deps.text_client is None or self.deps.text_model is None:
            return AgentResult(
                role=self.role_name,
                text=fallback_plan,
                confidence=0.82,
                artifacts={"queries": fallback_queries, "mode": "heuristic_fallback"},
            )

        prompt_tool_observations = []
        if self.deps.tools is not None:
            prompt_tool_observations = self.deps.tools.execute(
                context,
                self.deps.tools.default_calls_for_role(context, self.role_name),
                max_calls=2,
            )
        response = await self.deps.text_client.chat(
            model=self.deps.text_model,
            messages=[
                ModelMessage(role="system", content=PLANNER_SYSTEM_PROMPT),
                ModelMessage(role="user", content=self._render_prompt(context, prompt_tool_observations)),
            ],
            temperature=0.0,
            max_tokens=320,
            extra=self.deps.text_chat_extra or None,
        )
        payload = self._extract_json(response.text)
        if payload is None:
            plan_text, queries = fallback_plan, fallback_queries
            confidence = 0.62
            mode = "llm_parse_fallback"
        else:
            plan_text, queries = self._normalize_llm_plan(payload, context)
            confidence = 0.86
            mode = "llm"
        tool_observations = list(prompt_tool_observations)
        if self.deps.tools is not None and payload is not None:
            calls = normalize_tool_calls(
                payload.get("tool_calls"),
                allowed={"extract_key_terms", "inspect_inline_context", "list_answer_criteria"},
                max_calls=2,
            )
            if calls:
                tool_observations = self.deps.tools.execute(context, calls, max_calls=2)
                mode = f"{mode}_with_tools"
        return AgentResult(
            role=self.role_name,
            text=plan_text,
            confidence=confidence,
            artifacts={
                "queries": queries,
                "tool_observations": tool_observations,
                "usage": response.usage,
                "raw": response.raw,
                "mode": mode,
            },
            latency_ms=response.latency_ms,
        )

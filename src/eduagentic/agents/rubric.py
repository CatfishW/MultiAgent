from __future__ import annotations

import json
import re
from typing import Any

from ..core.contracts import AgentResult, ModelMessage
from ..prompts.templates import CRITERIA_SYSTEM_PROMPT
from .base import AgentContext, BaseAgent


class RubricAgent(BaseAgent):
    role_name = "rubric"

    def _fallback_summary(self, context: AgentContext) -> tuple[str, list[str], float]:
        rubric = context.example.rubric or []
        if not rubric:
            criteria = ["correctness", "clarity", "use evidence when available", "actionable next step"]
            summary = "No explicit criteria provided. Default to correctness, clarity, evidence use when available, and actionable next steps."
            confidence = 0.55
        else:
            criteria = [str(item).strip() for item in rubric[:10] if str(item).strip()]
            bullets = "\n".join(f"- {item}" for item in rubric[:10])
            summary = f"Prioritize the following criteria:\n{bullets}"
            confidence = 0.88
        return summary, criteria, confidence

    def _render_prompt(self, context: AgentContext) -> str:
        example = context.example
        parts = [
            f"Domain or dataset: {example.dataset_name}",
            f"Response regime: {context.route.regime.value}",
            f"Question or task:\n{example.question}",
        ]
        if example.rubric:
            parts.append("Explicit criteria:\n" + "\n".join(f"- {item}" for item in example.rubric[:12]))
        if example.context_text:
            parts.append(f"Inline context excerpt:\n{example.context_text[:1200]}")
        if example.dialogue_history:
            history = "\n".join(f"{turn.role}: {turn.text}" for turn in example.dialogue_history[-6:])
            parts.append(f"Recent interaction:\n{history}")
        parts.append("Produce reusable response criteria. Preserve explicit criteria exactly when possible.")
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

    @staticmethod
    def _criteria_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        criteria: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                criteria.append(text)
            if len(criteria) >= 10:
                break
        return criteria

    async def run(self, context: AgentContext) -> AgentResult:
        fallback_summary, fallback_criteria, fallback_confidence = self._fallback_summary(context)
        if self.deps.text_client is None or self.deps.text_model is None:
            return AgentResult(
                role=self.role_name,
                text=fallback_summary,
                confidence=fallback_confidence,
                artifacts={"criteria": fallback_criteria, "mode": "heuristic_fallback"},
            )

        response = await self.deps.text_client.chat(
            model=self.deps.text_model,
            messages=[
                ModelMessage(role="system", content=CRITERIA_SYSTEM_PROMPT),
                ModelMessage(role="user", content=self._render_prompt(context)),
            ],
            temperature=0.0,
            max_tokens=320,
            extra=self.deps.text_chat_extra or None,
        )
        payload = self._extract_json(response.text)
        if payload is None:
            summary = fallback_summary
            criteria = fallback_criteria
            confidence = 0.62
            mode = "llm_parse_fallback"
        else:
            criteria = self._criteria_list(payload.get("criteria")) or fallback_criteria
            summary = str(payload.get("summary") or "").strip()
            if not summary:
                summary = "Prioritize the following criteria:\n" + "\n".join(f"- {item}" for item in criteria)
            confidence = 0.86
            mode = "llm"
        return AgentResult(
            role=self.role_name,
            text=summary,
            confidence=confidence,
            artifacts={"criteria": criteria, "usage": response.usage, "raw": response.raw, "mode": mode},
            latency_ms=response.latency_ms,
        )

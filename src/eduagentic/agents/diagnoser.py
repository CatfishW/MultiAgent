from __future__ import annotations

import json
import re
from typing import Any

from ..core.contracts import AgentResult, ModelMessage, StudentState
from ..ml.student_state import StudentStateTracker
from ..prompts.templates import STATE_SYSTEM_PROMPT
from .base import AgentContext, BaseAgent


class DiagnoserAgent(BaseAgent):
    role_name = "diagnoser"

    def __init__(self, deps, tracker: StudentStateTracker | None = None) -> None:
        super().__init__(deps)
        self.tracker = tracker or StudentStateTracker()

    def _heuristic_result(self, context: AgentContext) -> tuple[StudentState, str]:
        state = self.tracker.infer(context.example)
        lines = [
            f"user_level: {state.level}",
            f"goals: {', '.join(state.goals) if state.goals else 'unspecified'}",
            f"style: {state.preferred_style or 'not stated'}",
        ]
        if state.misconceptions:
            lines.append("visible_state_notes:")
            lines.extend([f"- {item}" for item in state.misconceptions[:3]])
        return state, "\n".join(lines)

    def _render_prompt(self, context: AgentContext) -> str:
        example = context.example
        parts = [
            f"Domain or dataset: {example.dataset_name}",
            f"Question or task:\n{example.question}",
        ]
        if example.dialogue_history:
            history = "\n".join(f"{turn.role}: {turn.text}" for turn in example.dialogue_history[-8:])
            parts.append(f"Recent interaction:\n{history}")
        if example.context_text:
            parts.append(f"Inline context excerpt:\n{example.context_text[:1200]}")
        if example.rubric:
            parts.append("Explicit criteria:\n" + "\n".join(f"- {item}" for item in example.rubric[:8]))
        parts.append("Infer only visible user/task state. Do not infer demographics, identity, or persistent traits.")
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
    def _list_of_strings(value: Any, *, limit: int) -> list[str]:
        if not isinstance(value, list):
            return []
        items: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                items.append(text)
            if len(items) >= limit:
                break
        return items

    def _state_from_payload(self, payload: dict[str, Any], fallback: StudentState) -> StudentState:
        level = str(payload.get("level") or fallback.level or "unknown").strip().lower()
        if level not in {"beginner", "intermediate", "advanced", "unknown"}:
            level = fallback.level or "unknown"
        preferred_style = payload.get("preferred_style")
        preferred_style_text = str(preferred_style).strip() if preferred_style is not None else None
        if preferred_style_text not in {"step-by-step", "concise", "analogy"}:
            preferred_style_text = fallback.preferred_style
        goals = self._list_of_strings(payload.get("goals"), limit=6) or fallback.goals
        misconceptions = self._list_of_strings(payload.get("misconceptions"), limit=6) or fallback.misconceptions
        summary = str(payload.get("summary") or fallback.summary or f"level={level}").strip()
        return StudentState(
            level=level,
            goals=goals,
            misconceptions=misconceptions,
            strengths=fallback.strengths,
            preferred_style=preferred_style_text,
            summary=summary,
        )

    @staticmethod
    def _render_state(state: StudentState) -> str:
        lines = [
            f"user_level: {state.level}",
            f"goals: {', '.join(state.goals) if state.goals else 'unspecified'}",
            f"style: {state.preferred_style or 'not stated'}",
            f"summary: {state.summary or 'not stated'}",
        ]
        if state.misconceptions:
            lines.append("visible_state_notes:")
            lines.extend([f"- {item}" for item in state.misconceptions[:3]])
        return "\n".join(lines)

    async def run(self, context: AgentContext) -> AgentResult:
        fallback_state, fallback_text = self._heuristic_result(context)
        if self.deps.text_client is None or self.deps.text_model is None:
            return AgentResult(
                role=self.role_name,
                text=fallback_text,
                confidence=0.76,
                artifacts={"student_state": fallback_state, "task_state": fallback_state, "mode": "heuristic_fallback"},
            )

        response = await self.deps.text_client.chat(
            model=self.deps.text_model,
            messages=[
                ModelMessage(role="system", content=STATE_SYSTEM_PROMPT),
                ModelMessage(role="user", content=self._render_prompt(context)),
            ],
            temperature=0.0,
            max_tokens=280,
            extra=self.deps.text_chat_extra or None,
        )
        payload = self._extract_json(response.text)
        if payload is None:
            state = fallback_state
            text = fallback_text
            confidence = 0.58
            mode = "llm_parse_fallback"
        else:
            state = self._state_from_payload(payload, fallback_state)
            text = self._render_state(state)
            confidence = 0.84
            mode = "llm"
        return AgentResult(
            role=self.role_name,
            text=text,
            confidence=confidence,
            artifacts={"student_state": state, "task_state": state, "usage": response.usage, "raw": response.raw, "mode": mode},
            latency_ms=response.latency_ms,
        )

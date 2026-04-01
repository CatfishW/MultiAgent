from __future__ import annotations

from ..core.contracts import AgentResult
from .base import AgentContext, BaseAgent


class RubricAgent(BaseAgent):
    role_name = "rubric"

    async def run(self, context: AgentContext) -> AgentResult:
        rubric = context.example.rubric or []
        if not rubric:
            summary = "No explicit rubric provided. Default to correctness, clarity, scaffolding, and actionable feedback."
            confidence = 0.55
        else:
            bullets = "\n".join(f"- {item}" for item in rubric[:10])
            summary = f"Prioritize the following rubric dimensions:\n{bullets}"
            confidence = 0.88
        return AgentResult(role=self.role_name, text=summary, confidence=confidence)

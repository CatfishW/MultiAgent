from __future__ import annotations

from ..core.contracts import AgentResult
from ..ml.student_state import StudentStateTracker
from .base import AgentContext, BaseAgent


class DiagnoserAgent(BaseAgent):
    role_name = "diagnoser"

    def __init__(self, deps, tracker: StudentStateTracker | None = None) -> None:
        super().__init__(deps)
        self.tracker = tracker or StudentStateTracker()

    async def run(self, context: AgentContext) -> AgentResult:
        state = self.tracker.infer(context.example)
        lines = [
            f"student_level: {state.level}",
            f"goals: {', '.join(state.goals) if state.goals else 'unspecified'}",
            f"style: {state.preferred_style or 'not stated'}",
        ]
        if state.misconceptions:
            lines.append("possible_misconceptions:")
            lines.extend([f"- {item}" for item in state.misconceptions[:3]])
        return AgentResult(
            role=self.role_name,
            text="\n".join(lines),
            confidence=0.76,
            artifacts={"student_state": state},
        )

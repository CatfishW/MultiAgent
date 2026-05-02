from __future__ import annotations

from ..core.contracts import AgentResult, ModelMessage
from ..prompts.templates import CRITIC_SYSTEM_PROMPT
from ..utils.text import normalize_text
from .base import AgentContext, BaseAgent


class CriticAgent(BaseAgent):
    role_name = "critic"

    def _issues(self, context: AgentContext) -> list[str]:
        issues: list[str] = []
        draft = normalize_text(context.draft_answer or "")
        if context.retrieved_chunks and "[" not in (context.draft_answer or ""):
            issues.append("missing evidence markers despite retrieval-enabled answer")
        if context.example.rubric:
            for item in context.example.rubric[:4]:
                key = normalize_text(item)[:40]
                if key and key not in draft:
                    issues.append(f"may not fully cover rubric item: {item}")
        if context.student_state and context.student_state.level == "beginner" and len((context.draft_answer or "").split()) > 220:
            issues.append("too long for a beginner-facing answer")
        return issues

    async def run(self, context: AgentContext) -> AgentResult:
        issues = self._issues(context)
        confidence = max(0.15, 0.92 - 0.18 * len(issues))
        if self.deps.text_client is None or self.deps.text_model is None:
            return AgentResult(
                role=self.role_name,
                text=context.draft_answer or "",
                confidence=confidence,
                artifacts={"issues": issues, "mode": "heuristic_fallback"},
            )
        issue_block = "\n".join(f"- {item}" for item in issues) if issues else "- no deterministic issue found; verify briefly and preserve the answer if it is already correct"
        user_prompt = (
            f"Draft answer:\n{context.draft_answer}\n\n"
            f"Issues or checks:\n{issue_block}\n\n"
            "Return the final answer only. If the draft is already correct, return it unchanged."
        )
        response = await self.deps.text_client.chat(
            model=self.deps.text_model,
            messages=[
                ModelMessage(role="system", content=CRITIC_SYSTEM_PROMPT),
                ModelMessage(role="user", content=user_prompt),
            ],
            temperature=0.0,
            max_tokens=context.budget.max_response_tokens,
            extra=self.deps.text_chat_extra or None,
        )
        return AgentResult(
            role=self.role_name,
            text=response.text.strip(),
            confidence=confidence,
            artifacts={"issues": issues, "usage": response.usage, "raw": response.raw, "mode": "llm"},
            latency_ms=response.latency_ms,
        )

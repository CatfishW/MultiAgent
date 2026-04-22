from __future__ import annotations

from typing import Any

from ..core.contracts import AgentResult, ModelMessage
from ..prompts.templates import (
    EDUBENCH_JSON_SYSTEM_PROMPT,
    TUTOR_SYSTEM_PROMPT,
    VISION_TUTOR_SYSTEM_PROMPT,
)
from .base import AgentContext, BaseAgent


class TutorAgent(BaseAgent):
    role_name = "tutor"

    def _render_user_prompt(self, context: AgentContext) -> str:
        example = context.example
        parts = [f"Dataset: {example.dataset_name}", f"Question:\n{example.question}"]
        if example.choices:
            parts.append("Choices:\n" + "\n".join(f"- {choice}" for choice in example.choices))
        if example.dialogue_history:
            history = "\n".join(f"{turn.role}: {turn.text}" for turn in example.dialogue_history[-8:])
            parts.append(f"Recent dialogue:\n{history}")
        if context.student_state is not None:
            parts.append(f"Student profile:\n{context.student_state.summary}")
            if context.student_state.misconceptions:
                parts.append("Possible misconceptions:\n" + "\n".join(f"- {item}" for item in context.student_state.misconceptions[:3]))
        if context.plan_text:
            parts.append(f"Operating plan:\n{context.plan_text}")
        if context.rubric_summary:
            parts.append(f"Rubric:\n{context.rubric_summary}")
        if context.retrieved_chunks:
            evidence = "\n\n".join(
                f"[{chunk.doc_id}] {chunk.title}\n{chunk.text}" for chunk in context.retrieved_chunks
            )
            parts.append(f"Grounding evidence:\n{evidence}")
        elif context.example.context_text:
            parts.append(f"Inline context:\n{context.example.context_text}")
        parts.append(
            "Answer requirements:\n"
            "- be correct and pedagogically useful\n"
            "- be concise but not terse\n"
            "- include one clear next step or check-for-understanding when appropriate\n"
            "- when evidence is provided, cite with [doc_id]"
        )
        return "\n\n".join(parts)

    def _fallback_answer(self, context: AgentContext) -> str:
        if context.retrieved_chunks:
            lead = context.retrieved_chunks[0]
            return f"Based on {lead.title} [{lead.doc_id}], the key point is: {lead.text[:280]}"
        if context.example.context_text:
            return context.example.context_text[:320]
        return "I need either a configured local model or supporting context to answer this reliably."

    async def run(self, context: AgentContext) -> AgentResult:
        client = self.deps.vision_client if context.route.modality.value == "multimodal" and self.deps.vision_client else self.deps.text_client
        model = self.deps.vision_model if context.route.modality.value == "multimodal" and self.deps.vision_model else self.deps.text_model
        chat_extra = self.deps.vision_chat_extra if context.route.modality.value == "multimodal" else self.deps.text_chat_extra
        if client is None or model is None:
            answer = self._fallback_answer(context)
            return AgentResult(role=self.role_name, text=answer, confidence=0.4)

        eval_profile = str((context.example.metadata or {}).get("evaluation_profile", "")).lower()
        if eval_profile == "edubench_consensus":
            system_prompt = EDUBENCH_JSON_SYSTEM_PROMPT
        elif context.route.modality.value == "multimodal":
            system_prompt = VISION_TUTOR_SYSTEM_PROMPT
        else:
            system_prompt = TUTOR_SYSTEM_PROMPT
        messages = [
            ModelMessage(role="system", content=system_prompt),
            ModelMessage(role="user", content=self._render_user_prompt(context)),
        ]
        response = await client.chat(
            model=model,
            messages=messages,
            temperature=0.15,
            max_tokens=context.budget.max_response_tokens,
            images=context.example.images,
            extra=chat_extra or None,
        )
        return AgentResult(
            role=self.role_name,
            text=response.text.strip(),
            confidence=0.8,
            artifacts={"usage": response.usage, "model": response.model, "raw": response.raw},
            latency_ms=response.latency_ms,
        )

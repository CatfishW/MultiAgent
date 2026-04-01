from __future__ import annotations

from ..core.contracts import AgentResult, TaskRegime
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
                "Identify the student difficulty and the rubric dimensions.",
                "Explain what is correct, what is missing, and why.",
                "Give one or two high-value next steps, not a wall of feedback.",
            ]
        elif regime == TaskRegime.ADAPTIVE_TUTORING:
            steps = [
                "Infer student level and likely misconception.",
                "Choose the minimum explanation needed for progress.",
                "End with a check-for-understanding or next-step hint.",
            ]
        else:
            steps = [
                "Define the instructional goal and pacing constraints.",
                "Sequence activities from diagnosis to guided practice to check.",
                "Keep the plan lightweight and actionable.",
            ]
        if context.route.require_retrieval:
            steps.insert(1, "Use retrieval only for the parts that need grounding.")
        return "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(steps)), unique_queries[: context.budget.max_retrieval_queries]

    async def run(self, context: AgentContext) -> AgentResult:
        plan_text, queries = self._heuristic_plan(context)
        return AgentResult(
            role=self.role_name,
            text=plan_text,
            confidence=0.82,
            artifacts={"queries": queries},
        )

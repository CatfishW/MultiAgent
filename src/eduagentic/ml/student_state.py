from __future__ import annotations

from ..core.contracts import BenchmarkExample, ConversationTurn, StudentState
from ..utils.text import normalize_text


CONFUSION_CUES = ["i don't understand", "i am confused", "why", "stuck", "not sure", "doesn't make sense"]
ADVANCED_CUES = ["derive", "formal proof", "optimize", "counterexample", "generalize"]
BEGINNER_CUES = ["simple", "beginner", "basic", "step by step", "new to", "first time"]
STYLE_CUES = {
    "step-by-step": ["step by step", "walk me through", "show each step"],
    "concise": ["brief", "short answer", "concise"],
    "analogy": ["analogy", "example", "intuitive"],
}


class StudentStateTracker:
    def infer(self, example: BenchmarkExample) -> StudentState:
        text = "\n".join([turn.text for turn in example.dialogue_history] + [example.question])
        lowered = normalize_text(text)
        level = "intermediate"
        if any(cue in lowered for cue in BEGINNER_CUES):
            level = "beginner"
        elif any(cue in lowered for cue in ADVANCED_CUES):
            level = "advanced"

        misconceptions: list[str] = []
        for turn in example.dialogue_history:
            if turn.role != "user":
                continue
            line = normalize_text(turn.text)
            if "i thought" in line or "isn't it" in line or "so that means" in line:
                misconceptions.append(turn.text.strip())
        preferred_style = None
        for label, cues in STYLE_CUES.items():
            if any(cue in lowered for cue in cues):
                preferred_style = label
                break

        goals = []
        if "learn" in lowered or "understand" in lowered:
            goals.append("understand current task")
        if "solve" in lowered or "answer" in lowered:
            goals.append("solve current problem")
        if "plan" in lowered or "schedule" in lowered:
            goals.append("build action plan")

        confusion = any(cue in lowered for cue in CONFUSION_CUES)
        summary_parts = [f"level={level}"]
        if confusion:
            summary_parts.append("shows confusion")
        if misconceptions:
            summary_parts.append(f"possible misconceptions={len(misconceptions)}")
        if preferred_style:
            summary_parts.append(f"style={preferred_style}")
        return StudentState(
            level=level,
            goals=goals,
            misconceptions=misconceptions[:6],
            strengths=[],
            preferred_style=preferred_style,
            summary=", ".join(summary_parts),
        )

    def update_after_response(
        self,
        state: StudentState,
        user_turn: ConversationTurn,
        assistant_text: str,
    ) -> StudentState:
        lowered = normalize_text(user_turn.text)
        if "thanks" in lowered or "got it" in lowered:
            if "understand concept" in state.goals:
                state.strengths.append("accepted explanation")
        if "still confused" in lowered:
            state.misconceptions.append(user_turn.text)
        if assistant_text and len(assistant_text.split()) > 180 and state.preferred_style == "concise":
            state.preferred_style = "concise"
        return state


# Generic alias for non-educational deployments. The original class name stays
# available so existing datasets and experiment artifacts remain compatible.
TaskStateTracker = StudentStateTracker

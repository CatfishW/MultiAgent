from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Literal


class TaskRegime(str, Enum):
    EVIDENCE_GROUNDED = "evidence_grounded_reasoning"
    RUBRIC_FEEDBACK = "rubric_based_feedback"
    ADAPTIVE_TUTORING = "adaptive_tutoring"
    LESSON_PLANNING = "lesson_exercise_planning"
    # Generic aliases. The legacy values are kept so previous experiment
    # artifacts and dataset adapters remain readable.
    ADAPTIVE_RESPONSE = "adaptive_tutoring"
    TASK_PLANNING = "lesson_exercise_planning"


class ArchitectureFamily(str, Enum):
    CLASSICAL_RAG = "classical_rag"
    AGENTIC_RAG = "agentic_rag"
    NON_RAG_MULTI_AGENT = "non_rag_multi_agent"
    SINGLE_AGENT_NO_RAG = "single_agent_no_rag"
    HYBRID_FAST = "hybrid_fast"


class Modality(str, Enum):
    TEXT = "text"
    MULTIMODAL = "multimodal"


@dataclass(slots=True)
class BudgetPolicy:
    max_latency_ms: int = 12000
    max_tool_calls: int = 4
    max_agents: int = 4
    max_critic_rounds: int = 1
    max_retrieval_queries: int = 3
    max_context_chars: int = 6000
    max_response_tokens: int = 900


@dataclass(slots=True)
class ConversationTurn:
    role: Literal["user", "assistant", "system"]
    text: str


@dataclass(slots=True)
class BenchmarkExample:
    example_id: str
    dataset_name: str
    regime_hint: TaskRegime | None
    question: str
    gold_answer: str | None = None
    choices: list[str] | None = None
    context_text: str | None = None
    dialogue_history: list[ConversationTurn] = field(default_factory=list)
    rubric: list[str] | None = None
    images: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    reference_docs: list[dict[str, Any]] = field(default_factory=list)
    expected_doc_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ModelMessage:
    role: Literal["system", "user", "assistant"]
    content: Any


@dataclass(slots=True)
class ModelResponse:
    text: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] | None = None
    latency_ms: int | None = None


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    query_latency_ms: int | None = None


@dataclass(slots=True)
class StudentState:
    level: str = "unknown"
    goals: list[str] = field(default_factory=list)
    misconceptions: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    preferred_style: str | None = None
    summary: str = ""


# Generic alias for domain deployments outside education.
TaskState = StudentState


@dataclass(slots=True)
class RouteDecision:
    regime: TaskRegime
    architecture: ArchitectureFamily
    require_retrieval: bool
    use_critic: bool
    use_rubric_agent: bool
    modality: Modality = Modality.TEXT
    specialist_roles: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AgentResult:
    role: str
    text: str
    confidence: float = 0.5
    artifacts: dict[str, Any] = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)
    latency_ms: int | None = None


@dataclass(slots=True)
class TraceEvent:
    kind: str
    payload: dict[str, Any]


@dataclass(slots=True)
class PipelineResponse:
    answer: str
    architecture: ArchitectureFamily
    regime: TaskRegime
    route: RouteDecision
    citations: list[str] = field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    agent_outputs: dict[str, AgentResult] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    trace: list[TraceEvent] = field(default_factory=list)
    raw_model_outputs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationRecord:
    example_id: str
    dataset_name: str
    architecture: str
    metrics: dict[str, float]
    answer: str
    gold_answer: str | None
    retrieved_doc_ids: list[str] = field(default_factory=list)


def flatten_dialogue(turns: Iterable[ConversationTurn]) -> str:
    return "\n".join(f"{turn.role}: {turn.text}" for turn in turns)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from ..config import AppConfig
from ..core.contracts import ArchitectureFamily, BenchmarkExample, Modality, RouteDecision, TaskRegime
from ..utils.text import keyword_score, normalize_text


EVIDENCE_CUES = [
    "cite",
    "citation",
    "source",
    "evidence",
    "supporting facts",
    "reference",
    "grounded",
    "retrieval",
    "document",
    "chapter",
    "lecture",
    "verification",
]
COORDINATION_CUES = [
    "user",
    "profile",
    "state",
    "context",
    "criteria",
    "constraint",
    "workflow",
    "dependency",
    "rubric",
    "feedback",
    "student",
    "misconception",
    "hint",
    "next step",
    "lesson plan",
    "study plan",
    "adaptive",
    "scaffold",
    "dialogue",
    "pedagog",
]
RUBRIC_CUES = ["rubric", "criterion", "criteria", "score", "feedback", "comment"]
PLANNING_CUES = [
    "plan",
    "sequence",
    "schedule",
    "workflow",
    "roadmap",
    "steps",
    "next steps",
    "lesson",
    "curriculum",
    "study schedule",
    "exercise plan",
]
ADAPTATION_CUES = [
    "user",
    "profile",
    "beginner",
    "expert",
    "personalize",
    "adapt",
    "confused",
    "attempted",
    "hint",
    "misconception",
    "student",
    "explain",
    "teach",
    "tutor",
]

DATASET_PRIORS: dict[str, tuple[TaskRegime, ArchitectureFamily]] = {
    "hotpotqa": (TaskRegime.EVIDENCE_GROUNDED, ArchitectureFamily.AGENTIC_RAG),
    "fever": (TaskRegime.EVIDENCE_GROUNDED, ArchitectureFamily.CLASSICAL_RAG),
    "beir": (TaskRegime.EVIDENCE_GROUNDED, ArchitectureFamily.CLASSICAL_RAG),
    "wizard_of_wikipedia": (TaskRegime.EVIDENCE_GROUNDED, ArchitectureFamily.AGENTIC_RAG),
    "scienceqa": (TaskRegime.EVIDENCE_GROUNDED, ArchitectureFamily.AGENTIC_RAG),
    "tutoreval": (TaskRegime.ADAPTIVE_TUTORING, ArchitectureFamily.AGENTIC_RAG),
    "lm-science-tutor": (TaskRegime.ADAPTIVE_TUTORING, ArchitectureFamily.AGENTIC_RAG),
    "tutorbench": (TaskRegime.RUBRIC_FEEDBACK, ArchitectureFamily.NON_RAG_MULTI_AGENT),
    "mathtutorbench": (TaskRegime.RUBRIC_FEEDBACK, ArchitectureFamily.NON_RAG_MULTI_AGENT),
    "edubench": (TaskRegime.ADAPTIVE_TUTORING, ArchitectureFamily.HYBRID_FAST),
    "scrolls": (TaskRegime.EVIDENCE_GROUNDED, ArchitectureFamily.AGENTIC_RAG),
    "longbench-v2": (TaskRegime.EVIDENCE_GROUNDED, ArchitectureFamily.AGENTIC_RAG),
    "agentbench": (TaskRegime.LESSON_PLANNING, ArchitectureFamily.NON_RAG_MULTI_AGENT),
}


@dataclass(slots=True)
class TrainedRouter:
    vectorizer: CountVectorizer
    model: LogisticRegression


class LightweightRegimeRouter:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.trained: TrainedRouter | None = None
        if config.router.model_path and Path(config.router.model_path).exists():
            self.trained = self.load(config.router.model_path).trained

    def fit(self, texts: list[str], labels: list[str]) -> "LightweightRegimeRouter":
        vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1)
        X = vectorizer.fit_transform(texts)
        model = LogisticRegression(max_iter=1000, multi_class="auto")
        model.fit(X, labels)
        self.trained = TrainedRouter(vectorizer=vectorizer, model=model)
        return self

    def save(self, path: str | Path) -> Path:
        if self.trained is None:
            raise RuntimeError("Cannot save router before training or loading a model")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as fh:
            pickle.dump(self, fh)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "LightweightRegimeRouter":
        with Path(path).open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, LightweightRegimeRouter):
            raise TypeError(f"Unexpected router object: {type(obj)!r}")
        return obj

    def _text_blob(self, example: BenchmarkExample) -> str:
        parts = [example.question, example.context_text or ""]
        if example.rubric:
            parts.extend(example.rubric)
        if example.dialogue_history:
            parts.extend(turn.text for turn in example.dialogue_history)
        return "\n".join(parts)

    def _heuristic_scores(self, example: BenchmarkExample) -> dict[str, float]:
        blob = normalize_text(self._text_blob(example))
        evidence = keyword_score(blob, EVIDENCE_CUES)
        coordination = keyword_score(blob, COORDINATION_CUES)
        rubric = keyword_score(blob, RUBRIC_CUES) + (0.2 if example.rubric else 0.0)
        planning = keyword_score(blob, PLANNING_CUES)
        adaptation = keyword_score(blob, ADAPTATION_CUES) + min(len(example.dialogue_history) / 6.0, 0.5)
        if example.context_text:
            evidence += 0.08
        if example.images:
            evidence += 0.05
            coordination += 0.03
        dataset_key = example.dataset_name.lower()
        if getattr(self.config.router, "use_dataset_priors", False) and dataset_key in DATASET_PRIORS:
            prior_regime, _ = DATASET_PRIORS[dataset_key]
            if prior_regime == TaskRegime.EVIDENCE_GROUNDED:
                evidence += 0.15
            elif prior_regime == TaskRegime.RUBRIC_FEEDBACK:
                coordination += 0.15
                rubric += 0.15
            elif prior_regime == TaskRegime.ADAPTIVE_TUTORING:
                coordination += 0.15
                adaptation += 0.15
            else:
                coordination += 0.12
                planning += 0.18
        return {
            "evidence": min(1.0, evidence),
            "coordination": min(1.0, coordination),
            "rubric": min(1.0, rubric),
            "planning": min(1.0, planning),
            "adaptation": min(1.0, adaptation),
            "tutoring": min(1.0, adaptation),
        }

    def decide(self, example: BenchmarkExample) -> RouteDecision:
        dataset_key = example.dataset_name.lower()
        scores = self._heuristic_scores(example)
        router_cfg = self.config.router
        use_heuristic_only = bool(getattr(router_cfg, "use_heuristic_only", False))
        use_classifier_only = bool(getattr(router_cfg, "use_classifier_only", False))
        if use_classifier_only and self.trained is None:
            raise RuntimeError(
                "router.use_classifier_only is set but no trained classifier is loaded. "
                "Set router.model_path or clear use_classifier_only."
            )
        if self.trained is not None and not use_heuristic_only:
            vector = self.trained.vectorizer.transform([self._text_blob(example)])
            prediction = self.trained.model.predict(vector)[0]
            predicted_family = ArchitectureFamily(prediction)
        elif getattr(router_cfg, "use_dataset_priors", False) and dataset_key in DATASET_PRIORS and not use_heuristic_only:
            _, predicted_family = DATASET_PRIORS[dataset_key]
        else:
            if scores["evidence"] >= router_cfg.evidence_threshold and scores["coordination"] < 0.35:
                predicted_family = ArchitectureFamily.CLASSICAL_RAG
            elif scores["evidence"] >= 0.35 and scores["coordination"] >= router_cfg.coordination_threshold:
                predicted_family = ArchitectureFamily.AGENTIC_RAG
            elif scores["coordination"] >= 0.55 and scores["evidence"] < 0.28:
                predicted_family = ArchitectureFamily.NON_RAG_MULTI_AGENT
            else:
                predicted_family = ArchitectureFamily.HYBRID_FAST

        if example.regime_hint is not None:
            regime = example.regime_hint
        elif scores["planning"] >= max(scores["rubric"], scores["adaptation"], 0.42):
            regime = TaskRegime.LESSON_PLANNING
        elif scores["rubric"] >= max(scores["adaptation"], 0.35):
            regime = TaskRegime.RUBRIC_FEEDBACK
        elif scores["adaptation"] >= 0.35:
            regime = TaskRegime.ADAPTIVE_TUTORING
        else:
            regime = TaskRegime.EVIDENCE_GROUNDED

        modality = Modality.MULTIMODAL if example.images else Modality.TEXT
        require_retrieval = predicted_family in {ArchitectureFamily.CLASSICAL_RAG, ArchitectureFamily.AGENTIC_RAG}
        if predicted_family == ArchitectureFamily.HYBRID_FAST:
            gate = float(getattr(router_cfg, "hybrid_retrieval_gate", 0.35))
            require_retrieval = scores["evidence"] >= gate or regime == TaskRegime.EVIDENCE_GROUNDED
        use_rubric_agent = bool(example.rubric) or regime == TaskRegime.RUBRIC_FEEDBACK
        use_critic = self.config.pipeline.enable_critic and (
            require_retrieval or use_rubric_agent or regime in {TaskRegime.ADAPTIVE_TUTORING, TaskRegime.LESSON_PLANNING}
        )

        specialist_roles = ["tutor"]
        if regime in {TaskRegime.ADAPTIVE_TUTORING, TaskRegime.LESSON_PLANNING}:
            specialist_roles.append("diagnoser")
        if predicted_family in {ArchitectureFamily.AGENTIC_RAG, ArchitectureFamily.HYBRID_FAST}:
            specialist_roles.append("planner")
        if require_retrieval:
            specialist_roles.append("retriever")
        if use_rubric_agent:
            specialist_roles.append("rubric")
        if use_critic:
            specialist_roles.append("critic")

        notes = []
        if predicted_family == ArchitectureFamily.HYBRID_FAST:
            notes.append("mixed-regime task: using conditional retrieval instead of always-on RAG")
        if modality == Modality.MULTIMODAL:
            notes.append("vision-capable endpoint required")

        return RouteDecision(
            regime=regime,
            architecture=predicted_family,
            require_retrieval=require_retrieval,
            use_critic=use_critic,
            use_rubric_agent=use_rubric_agent,
            modality=modality,
            specialist_roles=specialist_roles,
            scores=scores,
            notes=notes,
        )

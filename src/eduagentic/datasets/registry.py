from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

from ..config import AppConfig
from ..core.contracts import BenchmarkExample, TaskRegime
from .adapters import (
    HuggingFaceAdapter,
    LocalJsonAdapter,
    LocalJsonlAdapter,
    TRANSFORMS,
    generic_text_transform,
)
from .base import DatasetAdapter, DatasetSpec


class DatasetRegistry:
    """Registry covering every benchmark family named in the paper.

    Some datasets are wired to public Hugging Face defaults when commonly
    available. Others expose a local-schema adapter by default because the
    public artifact is a repo snapshot, a gated resource, or a benchmark with
    custom execution environments.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.specs = self._build_specs()

    def _build_specs(self) -> dict[str, DatasetSpec]:
        return {
            "EduBench": DatasetSpec(
                name="EduBench",
                family="education_specific",
                regime=TaskRegime.ADAPTIVE_TUTORING,
                default_loader="huggingface",
                default_source="DirectionAI/EduBench",
                notes="Broad educational benchmark spanning student- and teacher-facing scenarios.",
            ),
            "TutorEval": DatasetSpec(
                name="TutorEval",
                family="education_specific",
                regime=TaskRegime.ADAPTIVE_TUTORING,
                default_loader="huggingface",
                default_source="princeton-nlp/TutorEval",
                notes="Science tutoring QA with open-book and closed-book settings.",
            ),
            "LM-Science-Tutor": DatasetSpec(
                name="LM-Science-Tutor",
                family="education_specific",
                regime=TaskRegime.ADAPTIVE_TUTORING,
                default_loader="huggingface",
                default_source="princeton-nlp/TutorEval",
                notes="Alias mapping for TutorEval / LM-Science-Tutor experiments.",
            ),
            "ScienceQA": DatasetSpec(
                name="ScienceQA",
                family="education_specific",
                regime=TaskRegime.EVIDENCE_GROUNDED,
                default_loader="huggingface",
                default_source="derek-thomas/ScienceQA",
                modality="multimodal",
                notes="Multimodal science QA with lecture-style context.",
            ),
            "MathTutorBench": DatasetSpec(
                name="MathTutorBench",
                family="education_specific",
                regime=TaskRegime.RUBRIC_FEEDBACK,
                default_loader="local_jsonl",
                default_source=None,
                notes="Local adapter expected because the benchmark commonly ships as repo assets / task files.",
                requires_local_assets=True,
            ),
            "TutorBench": DatasetSpec(
                name="TutorBench",
                family="education_specific",
                regime=TaskRegime.RUBRIC_FEEDBACK,
                default_loader="huggingface",
                default_source="tutorbench/tutorbench",
                notes="Tutoring and rubric-heavy benchmark.",
            ),
            "HotpotQA": DatasetSpec(
                name="HotpotQA",
                family="transferable",
                regime=TaskRegime.EVIDENCE_GROUNDED,
                default_loader="huggingface",
                default_source="hotpotqa/hotpot_qa",
                notes="Multi-hop retrieval with supporting-fact supervision.",
            ),
            "AgentBench": DatasetSpec(
                name="AgentBench",
                family="transferable",
                regime=TaskRegime.LESSON_PLANNING,
                default_loader="local_jsonl",
                default_source=None,
                notes="Execution environments vary, so a local normalized export is the stable default.",
                requires_local_assets=True,
            ),
            "SCROLLS": DatasetSpec(
                name="SCROLLS",
                family="transferable",
                regime=TaskRegime.EVIDENCE_GROUNDED,
                default_loader="huggingface",
                default_source="tau/scrolls",
                notes="Long-context stress suite.",
            ),
            "LongBench-v2": DatasetSpec(
                name="LongBench-v2",
                family="transferable",
                regime=TaskRegime.EVIDENCE_GROUNDED,
                default_loader="huggingface",
                default_source="zai-org/LongBench-v2",
                notes="Realistic long-context multitask benchmark.",
            ),
            "BEIR": DatasetSpec(
                name="BEIR",
                family="transferable",
                regime=TaskRegime.EVIDENCE_GROUNDED,
                default_loader="local_jsonl",
                default_source=None,
                notes="Recommended workflow: export the target BEIR subset into normalized JSONL for repeatable runs.",
                requires_local_assets=True,
            ),
            "FEVER": DatasetSpec(
                name="FEVER",
                family="transferable",
                regime=TaskRegime.EVIDENCE_GROUNDED,
                default_loader="huggingface",
                default_source="fever/fever",
                notes="Fact verification with evidence.",
            ),
            "Wizard of Wikipedia": DatasetSpec(
                name="Wizard of Wikipedia",
                family="transferable",
                regime=TaskRegime.EVIDENCE_GROUNDED,
                default_loader="huggingface",
                default_source="chujiezheng/wizard_of_wikipedia",
                notes="Knowledge-grounded dialogue benchmark.",
            ),
        }

    def _apply_override(self, spec: DatasetSpec) -> DatasetSpec:
        override = self.config.datasets.registry_overrides.get(spec.name, {})
        if not override:
            return spec
        return replace(spec, **override)

    def names(self) -> list[str]:
        return list(self.specs.keys())

    def adapter_for(self, dataset_name: str) -> DatasetAdapter:
        try:
            spec = self._apply_override(self.specs[dataset_name])
        except KeyError as exc:
            raise KeyError(f"Unknown dataset: {dataset_name}. Available: {', '.join(self.names())}") from exc

        transform = generic_text_transform
        if spec.name == "HotpotQA":
            transform = TRANSFORMS["hotpot"]
        elif spec.name == "FEVER":
            transform = TRANSFORMS["fever"]
        elif spec.name == "ScienceQA":
            transform = TRANSFORMS["scienceqa"]
        elif spec.name == "Wizard of Wikipedia":
            transform = TRANSFORMS["wizard"]
        elif spec.name in {"SCROLLS", "LongBench-v2"}:
            transform = TRANSFORMS["long_context"]

        loader = spec.default_loader
        if loader == "huggingface":
            return HuggingFaceAdapter(spec, spec.default_source or "", transform=transform, subset=spec.subset)
        if loader == "local_json":
            return LocalJsonAdapter(spec, transform=transform)
        return LocalJsonlAdapter(spec, transform=transform)

    def load(
        self,
        dataset_name: str,
        *,
        source: str | None = None,
        split: str = "test",
        limit: int | None = None,
    ) -> list[BenchmarkExample]:
        adapter = self.adapter_for(dataset_name)
        if source is None:
            spec = self._apply_override(self.specs[dataset_name])
            source = spec.default_source
        return adapter.load(source=source, split=split, limit=limit)

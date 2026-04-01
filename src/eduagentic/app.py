from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import AppConfig, load_app_config
from .core.contracts import ArchitectureFamily, BenchmarkExample, PipelineResponse, TaskRegime
from .datasets.registry import DatasetRegistry
from .evaluation.evaluator import BenchmarkEvaluator
from .llm import ModelRegistry
from .ml.regime_router import LightweightRegimeRouter
from .ml.student_state import StudentStateTracker
from .orchestration.pipelines import AgenticRAGPipeline, ClassicalRAGPipeline, HybridFastPipeline, MultiAgentNoRAGPipeline, SingleAgentNoRAGPipeline
from .retrieval import ContextPacker, HybridIndex, LightweightReranker, chunk_documents, load_documents_from_path
from .agents.base import AgentDependencies


class ConferenceEduSystem:
    def __init__(self, config: str | Path | dict[str, Any] | AppConfig | None = None) -> None:
        self.config = load_app_config(config)
        self.registry = ModelRegistry(self.config)
        self.router = LightweightRegimeRouter(self.config)
        self.state_tracker = StudentStateTracker()
        self.dataset_registry = DatasetRegistry(self.config)
        self.evaluator = BenchmarkEvaluator()
        self._deps = AgentDependencies(
            retriever=None,
            reranker=LightweightReranker(),
            packer=ContextPacker(max_chars=self.config.budget.max_context_chars, mmr_lambda=self.config.retriever.mmr_lambda),
        )
        self._pipelines = None

    async def initialize_models(self) -> None:
        await self.registry.refresh(force=True)
        if "llm" in self.config.endpoints and self.config.endpoints["llm"].enabled:
            descriptor = await self.registry.pick_model(capability="text", endpoint_name="llm")
            self._deps.text_model = descriptor.model_id
            self._deps.text_client = self.registry.client_for(self.config.endpoints["llm"])
            self._deps.text_chat_extra = dict(self.config.endpoints["llm"].chat_extra)
        if "mllm" in self.config.endpoints and self.config.endpoints["mllm"].enabled:
            try:
                descriptor = await self.registry.pick_model(capability="multimodal", endpoint_name="mllm")
                self._deps.vision_model = descriptor.model_id
                self._deps.vision_client = self.registry.client_for(self.config.endpoints["mllm"])
                self._deps.vision_chat_extra = dict(self.config.endpoints["mllm"].chat_extra)
            except LookupError:
                self._deps.vision_model = None
                self._deps.vision_client = None
        self._pipelines = {
            ArchitectureFamily.CLASSICAL_RAG: ClassicalRAGPipeline(self.config, self._deps, tracker=self.state_tracker),
            ArchitectureFamily.AGENTIC_RAG: AgenticRAGPipeline(self.config, self._deps, tracker=self.state_tracker),
            ArchitectureFamily.NON_RAG_MULTI_AGENT: MultiAgentNoRAGPipeline(self.config, self._deps, tracker=self.state_tracker),
            ArchitectureFamily.SINGLE_AGENT_NO_RAG: SingleAgentNoRAGPipeline(self.config, self._deps, tracker=self.state_tracker),
            ArchitectureFamily.HYBRID_FAST: HybridFastPipeline(self.config, self._deps, tracker=self.state_tracker),
        }

    def index_documents(self, docs_or_path) -> HybridIndex:
        if isinstance(docs_or_path, (str, Path)):
            documents = load_documents_from_path(docs_or_path)
        else:
            documents = list(docs_or_path)
        chunks = chunk_documents(
            documents,
            chunk_size=self.config.retriever.chunk_size,
            chunk_overlap=self.config.retriever.chunk_overlap,
        )
        index = HybridIndex(self.config.retriever).fit(chunks)
        self._deps.retriever = index
        if self._pipelines is not None:
            for pipeline in self._pipelines.values():
                pipeline.deps.retriever = index
        return index

    async def run_example(
        self,
        example: BenchmarkExample,
        *,
        architecture: ArchitectureFamily | str | None = None,
    ) -> PipelineResponse:
        if self._pipelines is None:
            await self.initialize_models()
        route = self.router.decide(example)
        if architecture is not None:
            route.architecture = ArchitectureFamily(architecture)
        pipeline = self._pipelines[route.architecture]
        return await pipeline.run(example, route)

    async def answer(
        self,
        question: str,
        *,
        dataset_name: str = "custom",
        context_text: str | None = None,
        rubric: list[str] | None = None,
        images: list[str] | None = None,
        regime_hint: TaskRegime | None = None,
        architecture: ArchitectureFamily | str | None = None,
    ) -> PipelineResponse:
        example = BenchmarkExample(
            example_id="custom-0",
            dataset_name=dataset_name,
            regime_hint=regime_hint,
            question=question,
            context_text=context_text,
            rubric=rubric,
            images=images,
        )
        return await self.run_example(example, architecture=architecture)

    def load_examples(self, dataset_name: str, *, source: str | None = None, split: str = "test", limit: int | None = None) -> list[BenchmarkExample]:
        return self.dataset_registry.load(dataset_name, source=source, split=split, limit=limit)

    async def evaluate_dataset(
        self,
        dataset_name: str,
        *,
        source: str | None = None,
        split: str = "test",
        architecture: ArchitectureFamily | str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        examples = self.load_examples(dataset_name, source=source, split=split, limit=limit)
        return await self.evaluator.evaluate(self, examples, architecture=architecture)

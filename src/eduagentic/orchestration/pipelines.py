from __future__ import annotations

from dataclasses import replace
import time
from typing import Any

from ..agents import AgentContext, AgentDependencies, CriticAgent, DiagnoserAgent, PlannerAgent, RetrieverAgent, RubricAgent, TutorAgent
from ..config import AppConfig
from ..core.contracts import (
    AgentResult,
    ArchitectureFamily,
    BenchmarkExample,
    PipelineResponse,
    RouteDecision,
    TaskRegime,
    TraceEvent,
)
from ..ml.student_state import StudentStateTracker
from .runtime import FastGraphRuntime
from .swarm_bridge import SwarmRuntimeAdapter


class BasePipeline:
    architecture = ArchitectureFamily.HYBRID_FAST

    def __init__(self, config: AppConfig, deps: AgentDependencies, tracker: StudentStateTracker | None = None) -> None:
        self.config = config
        self.deps = deps
        self.runtime = FastGraphRuntime()
        self.tracker = tracker or StudentStateTracker()
        self.planner = PlannerAgent(deps)
        self.diagnoser = DiagnoserAgent(deps, tracker=self.tracker)
        self.retriever = RetrieverAgent(deps)
        self.rubric = RubricAgent(deps)
        self.tutor = TutorAgent(deps)
        self.critic = CriticAgent(deps)

    def _empty_context(self, example: BenchmarkExample, route: RouteDecision) -> AgentContext:
        return AgentContext(example=example, route=route, budget=self.config.budget)

    def _record(self, trace: list[TraceEvent], kind: str, **payload) -> None:
        trace.append(TraceEvent(kind=kind, payload=payload))

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _usage_token_counts(self, usage: dict[str, Any]) -> tuple[float, float, float]:
        prompt_tokens = self._safe_float(
            usage.get("prompt_tokens", usage.get("input_tokens", usage.get("prompt_token_count", 0.0))),
            0.0,
        )
        completion_tokens = self._safe_float(
            usage.get("completion_tokens", usage.get("output_tokens", usage.get("completion_token_count", 0.0))),
            0.0,
        )
        total_tokens = self._safe_float(usage.get("total_tokens", usage.get("total_token_count", 0.0)), 0.0)
        if total_tokens <= 0:
            total_tokens = prompt_tokens + completion_tokens
        return prompt_tokens, completion_tokens, total_tokens

    def _finalize(
        self,
        *,
        example: BenchmarkExample,
        route: RouteDecision,
        answer_result: AgentResult,
        retrieved_chunks,
        agent_outputs: dict[str, AgentResult],
        trace: list[TraceEvent],
        started: float,
    ) -> PipelineResponse:
        total_latency_ms = float(int((time.perf_counter() - started) * 1000))
        llm_call_count = 0.0
        prompt_tokens = 0.0
        completion_tokens = 0.0
        total_tokens = 0.0
        api_time_ms = 0.0
        agent_time_ms = 0.0
        retrieval_query_count = 0.0

        for output in agent_outputs.values():
            if output.latency_ms is not None:
                agent_time_ms += max(0.0, self._safe_float(output.latency_ms, 0.0))

            artifacts = output.artifacts if isinstance(output.artifacts, dict) else {}
            usage = artifacts.get("usage")
            if isinstance(usage, dict):
                llm_call_count += 1.0
                if output.latency_ms is not None:
                    api_time_ms += max(0.0, self._safe_float(output.latency_ms, 0.0))
                p_tokens, c_tokens, t_tokens = self._usage_token_counts(usage)
                prompt_tokens += p_tokens
                completion_tokens += c_tokens
                total_tokens += t_tokens

            queries = artifacts.get("queries")
            if isinstance(queries, list):
                retrieval_query_count += float(len([query for query in queries if str(query).strip()]))

        retrieved_chunk_count = float(len(retrieved_chunks))
        trace_event_count = float(len(trace))
        complexity_units = (
            total_tokens
            + (retrieval_query_count * 160.0)
            + (retrieved_chunk_count * 90.0)
            + (float(len(agent_outputs)) * 240.0)
            + (trace_event_count * 30.0)
        )
        non_api_time_ms = max(0.0, total_latency_ms - api_time_ms)
        api_time_ratio = (api_time_ms / total_latency_ms) if total_latency_ms > 0 else 0.0
        complexity_per_second = complexity_units / max(0.001, total_latency_ms / 1000.0)

        metrics = {
            "latency_ms": total_latency_ms,
            "api_time_ms": api_time_ms,
            "non_api_time_ms": non_api_time_ms,
            "api_time_ratio": api_time_ratio,
            "agent_time_ms": agent_time_ms,
            "agent_count": float(len(agent_outputs)),
            "llm_call_count": llm_call_count,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "retrieval_query_count": retrieval_query_count,
            "retrieved_chunks": retrieved_chunk_count,
            "trace_event_count": trace_event_count,
            "complexity_units": complexity_units,
            "complexity_per_second": complexity_per_second,
        }
        citations = []
        for output in agent_outputs.values():
            for citation in output.citations:
                if citation not in citations:
                    citations.append(citation)
        if not citations:
            citations = [chunk.doc_id for chunk in retrieved_chunks]
        return PipelineResponse(
            answer=answer_result.text,
            architecture=self.architecture,
            regime=route.regime,
            route=route,
            citations=citations,
            retrieved_chunks=list(retrieved_chunks),
            agent_outputs=agent_outputs,
            metrics=metrics,
            trace=trace,
            raw_model_outputs={
                name: output.artifacts.get("raw")
                for name, output in agent_outputs.items()
                if output.artifacts.get("raw") is not None
            },
        )


class ClassicalRAGPipeline(BasePipeline):
    architecture = ArchitectureFamily.CLASSICAL_RAG

    async def run(self, example: BenchmarkExample, route: RouteDecision) -> PipelineResponse:
        started = time.perf_counter()
        trace: list[TraceEvent] = []
        agent_outputs: dict[str, AgentResult] = {}
        context = self._empty_context(example, route)
        if self.deps.retriever is not None:
            retrieval = await self.retriever.run(context)
            agent_outputs["retriever"] = retrieval
            context = replace(context, retrieved_chunks=retrieval.artifacts.get("chunks", []))
            self._record(trace, "retrieval", queries=retrieval.artifacts.get("queries", []), count=len(context.retrieved_chunks))
        tutor = await self.tutor.run(context)
        agent_outputs["tutor"] = tutor
        answer = tutor
        if route.use_critic:
            critic = await self.critic.run(replace(context, draft_answer=tutor.text))
            agent_outputs["critic"] = critic
            answer = critic
        return self._finalize(
            example=example,
            route=route,
            answer_result=answer,
            retrieved_chunks=context.retrieved_chunks,
            agent_outputs=agent_outputs,
            trace=trace,
            started=started,
        )


class AgenticRAGPipeline(BasePipeline):
    architecture = ArchitectureFamily.AGENTIC_RAG

    async def _prepare_context(self, example: BenchmarkExample, route: RouteDecision) -> tuple[AgentContext, dict[str, AgentResult], list[TraceEvent]]:
        trace: list[TraceEvent] = []
        base_context = self._empty_context(example, route)
        tasks = {
            "planner": lambda: self.planner.run(base_context),
        }
        if self.config.pipeline.enable_diagnoser:
            tasks["diagnoser"] = lambda: self.diagnoser.run(base_context)
        if route.use_rubric_agent:
            tasks["rubric"] = lambda: self.rubric.run(base_context)
        if self.config.pipeline.enable_swarm_runtime and len(tasks) > 1:
            swarm = SwarmRuntimeAdapter()
            try:
                initial = await swarm.run_parallel_roles(tasks)
            finally:
                swarm.close()
        else:
            initial = await self.runtime.run_parallel(tasks)
        student_state = initial.get("diagnoser", AgentResult(role="diagnoser", text="", artifacts={})).artifacts.get("student_state")
        search_queries = initial.get("planner", AgentResult(role="planner", text="", artifacts={})).artifacts.get("queries", [example.question])
        rubric_summary = initial.get("rubric").text if "rubric" in initial else None
        context = replace(
            base_context,
            student_state=student_state,
            plan_text=initial.get("planner").text if "planner" in initial else None,
            search_queries=search_queries,
            rubric_summary=rubric_summary,
        )
        self._record(trace, "pre_tutor", roles=list(initial.keys()))
        return context, initial, trace

    async def run(self, example: BenchmarkExample, route: RouteDecision) -> PipelineResponse:
        started = time.perf_counter()
        context, agent_outputs, trace = await self._prepare_context(example, route)
        retrieval = await self.retriever.run(context)
        agent_outputs["retriever"] = retrieval
        context = replace(context, retrieved_chunks=retrieval.artifacts.get("chunks", []))
        self._record(trace, "retrieval", queries=context.search_queries, count=len(context.retrieved_chunks))
        tutor = await self.tutor.run(context)
        agent_outputs["tutor"] = tutor
        answer = tutor
        if route.use_critic:
            critic = await self.critic.run(replace(context, draft_answer=tutor.text))
            agent_outputs["critic"] = critic
            answer = critic
        return self._finalize(
            example=example,
            route=route,
            answer_result=answer,
            retrieved_chunks=context.retrieved_chunks,
            agent_outputs=agent_outputs,
            trace=trace,
            started=started,
        )


class MultiAgentNoRAGPipeline(BasePipeline):
    architecture = ArchitectureFamily.NON_RAG_MULTI_AGENT

    async def run(self, example: BenchmarkExample, route: RouteDecision) -> PipelineResponse:
        started = time.perf_counter()
        trace: list[TraceEvent] = []
        base_context = self._empty_context(example, route)
        tasks = {
            "planner": lambda: self.planner.run(base_context),
        }
        if self.config.pipeline.enable_diagnoser:
            tasks["diagnoser"] = lambda: self.diagnoser.run(base_context)
        if route.use_rubric_agent:
            tasks["rubric"] = lambda: self.rubric.run(base_context)
        initial = await self.runtime.run_parallel(tasks)
        context = replace(
            base_context,
            student_state=initial.get("diagnoser", AgentResult(role="diagnoser", text="", artifacts={})).artifacts.get("student_state"),
            plan_text=initial.get("planner").text if "planner" in initial else None,
            rubric_summary=initial.get("rubric").text if "rubric" in initial else None,
        )
        self._record(trace, "pre_tutor", roles=list(initial.keys()))
        tutor = await self.tutor.run(context)
        initial["tutor"] = tutor
        answer = tutor
        if route.use_critic:
            critic = await self.critic.run(replace(context, draft_answer=tutor.text))
            initial["critic"] = critic
            answer = critic
        return self._finalize(
            example=example,
            route=route,
            answer_result=answer,
            retrieved_chunks=[],
            agent_outputs=initial,
            trace=trace,
            started=started,
        )


class SingleAgentNoRAGPipeline(BasePipeline):
    architecture = ArchitectureFamily.SINGLE_AGENT_NO_RAG

    async def run(self, example: BenchmarkExample, route: RouteDecision) -> PipelineResponse:
        started = time.perf_counter()
        trace: list[TraceEvent] = []
        context = self._empty_context(example, route)
        tutor = await self.tutor.run(context)
        agent_outputs: dict[str, AgentResult] = {"tutor": tutor}
        answer = tutor
        if route.use_critic:
            critic = await self.critic.run(replace(context, draft_answer=tutor.text))
            agent_outputs["critic"] = critic
            answer = critic
        return self._finalize(
            example=example,
            route=route,
            answer_result=answer,
            retrieved_chunks=[],
            agent_outputs=agent_outputs,
            trace=trace,
            started=started,
        )


class HybridFastPipeline(BasePipeline):
    architecture = ArchitectureFamily.HYBRID_FAST

    async def run(self, example: BenchmarkExample, route: RouteDecision) -> PipelineResponse:
        started = time.perf_counter()
        trace: list[TraceEvent] = []
        base_context = self._empty_context(example, route)
        tasks = {"planner": lambda: self.planner.run(base_context)}
        if self.config.pipeline.enable_diagnoser:
            tasks["diagnoser"] = lambda: self.diagnoser.run(base_context)
        if route.use_rubric_agent:
            tasks["rubric"] = lambda: self.rubric.run(base_context)
        initial = await self.runtime.run_parallel(tasks)
        context = replace(
            base_context,
            student_state=initial.get("diagnoser", AgentResult(role="diagnoser", text="", artifacts={})).artifacts.get("student_state"),
            plan_text=initial.get("planner").text if "planner" in initial else None,
            search_queries=initial.get("planner", AgentResult(role="planner", text="", artifacts={})).artifacts.get("queries", [example.question]),
            rubric_summary=initial.get("rubric").text if "rubric" in initial else None,
        )
        agent_outputs = dict(initial)
        if route.require_retrieval and self.deps.retriever is not None:
            retrieval = await self.retriever.run(context)
            agent_outputs["retriever"] = retrieval
            context = replace(context, retrieved_chunks=retrieval.artifacts.get("chunks", []))
            self._record(trace, "retrieval", queries=context.search_queries, count=len(context.retrieved_chunks))
        tutor = await self.tutor.run(context)
        agent_outputs["tutor"] = tutor
        answer = tutor
        if not context.retrieved_chunks and self.deps.retriever is not None and route.scores.get("evidence", 0.0) >= 0.45:
            retrieval = await self.retriever.run(replace(context, search_queries=[example.question]))
            if retrieval.artifacts.get("chunks"):
                agent_outputs["retriever_fallback"] = retrieval
                context = replace(context, retrieved_chunks=retrieval.artifacts.get("chunks", []))
                tutor = await self.tutor.run(context)
                agent_outputs["tutor_fallback"] = tutor
                answer = tutor
                self._record(trace, "retrieval_fallback", count=len(context.retrieved_chunks))
        if route.use_critic:
            critic = await self.critic.run(replace(context, draft_answer=answer.text))
            agent_outputs["critic"] = critic
            answer = critic
        return self._finalize(
            example=example,
            route=route,
            answer_result=answer,
            retrieved_chunks=context.retrieved_chunks,
            agent_outputs=agent_outputs,
            trace=trace,
            started=started,
        )

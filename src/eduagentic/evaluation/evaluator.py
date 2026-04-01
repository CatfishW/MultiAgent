from __future__ import annotations

from typing import Any

from ..core.contracts import ArchitectureFamily, BenchmarkExample, EvaluationRecord
from .metrics import canonical_answer_text, compute_metrics, summarize


class BenchmarkEvaluator:
    async def evaluate(
        self,
        system,
        examples: list[BenchmarkExample],
        *,
        architecture: ArchitectureFamily | str | None = None,
    ) -> dict[str, Any]:
        records: list[EvaluationRecord] = []
        metric_rows: list[dict[str, float]] = []
        for example in examples:
            response = await system.run_example(example, architecture=architecture)
            normalized_answer = canonical_answer_text(response.answer)
            metrics = compute_metrics(example, response)
            metric_rows.append(metrics)
            records.append(
                EvaluationRecord(
                    example_id=example.example_id,
                    dataset_name=example.dataset_name,
                    architecture=response.architecture.value,
                    metrics=metrics,
                    answer=normalized_answer,
                    gold_answer=example.gold_answer,
                    retrieved_doc_ids=[chunk.doc_id for chunk in response.retrieved_chunks],
                )
            )
        summary = summarize(metric_rows)
        return {
            "count": len(records),
            "summary": summary,
            "records": records,
        }

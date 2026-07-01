from math import ceil
from statistics import mean
from time import perf_counter
from uuid import UUID

from tutor_bot.evaluation.retrieval_evaluation_report import RetrievalEvaluationReport
from tutor_bot.retrieval.hybrid_search_service import HybridSearchService
from tutor_bot.retrieval.reranker_context_gate import RerankerContextGate
from tutor_bot.schemas.golden_dataset import GoldenDataset


class RetrievalEvaluator:
    def __init__(
        self,
        search_service: HybridSearchService,
        context_gate: RerankerContextGate,
        retrieval_limit: int = 10,
        recall_k: int = 5,
    ) -> None:
        if retrieval_limit <= 0 or recall_k <= 0:
            raise ValueError("Retrieval limit and recall K must be positive")

        if recall_k > retrieval_limit:
            raise ValueError("Recall K cannot exceed retrieval limit")

        self._search_service = search_service
        self._context_gate = context_gate
        self._retrieval_limit = retrieval_limit
        self._recall_k = recall_k

    def evaluate(
        self,
        dataset: GoldenDataset,
    ) -> RetrievalEvaluationReport:
        recalls = []
        reciprocal_ranks = []
        context_gate_matches = []
        latencies_ms = []

        for retrieval_case in dataset.retrieval_cases:
            started_at = perf_counter()
            results = self._search_service.search(
                retrieval_case.question,
                limit=self._retrieval_limit,
            )
            latencies_ms.append((perf_counter() - started_at) * 1000)

            context_result = self._context_gate.select(results)
            context_gate_matches.append(
                context_result.has_sufficient_context == retrieval_case.should_find_answer
            )

            if not retrieval_case.should_find_answer:
                continue

            expected_note_ids = {UUID(note_id) for note_id in retrieval_case.expected_note_ids}
            top_note_ids = {result.chunk.note_id for result in results[: self._recall_k]}
            recalls.append(len(expected_note_ids & top_note_ids) / len(expected_note_ids))

            first_expected_rank = next(
                (
                    rank
                    for rank, result in enumerate(results, start=1)
                    if result.chunk.note_id in expected_note_ids
                ),
                None,
            )
            reciprocal_ranks.append(
                0.0 if first_expected_rank is None else 1.0 / first_expected_rank
            )

        return RetrievalEvaluationReport(
            case_count=len(dataset.retrieval_cases),
            answerable_case_count=len(recalls),
            recall_k=self._recall_k,
            recall_at_k=self._mean_or_zero(recalls),
            mean_reciprocal_rank=self._mean_or_zero(reciprocal_ranks),
            context_gate_accuracy=mean(context_gate_matches),
            mean_latency_ms=mean(latencies_ms),
            p95_latency_ms=self._percentile_95(latencies_ms),
        )

    def _mean_or_zero(
        self,
        values: list[float],
    ) -> float:
        if not values:
            return 0.0

        return mean(values)

    def _percentile_95(
        self,
        values: list[float],
    ) -> float:
        sorted_values = sorted(values)
        percentile_index = ceil(len(sorted_values) * 0.95) - 1

        return sorted_values[percentile_index]

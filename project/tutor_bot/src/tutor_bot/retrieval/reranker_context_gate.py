from collections.abc import Sequence

from tutor_bot.retrieval.context_gate_result import ContextGateResult
from tutor_bot.retrieval.hybrid_search_result import HybridSearchResult


class RerankerContextGate:
    def __init__(
        self,
        minimum_reranker_score: float,
        context_limit: int = 5,
    ) -> None:
        if context_limit <= 0:
            raise ValueError("Context limit must be positive")

        self._minimum_reranker_score = minimum_reranker_score
        self._context_limit = context_limit

    def select(
        self,
        results: Sequence[HybridSearchResult],
    ) -> ContextGateResult:
        selected_results = tuple(
            result for result in results if result.reranker_score >= self._minimum_reranker_score
        )[: self._context_limit]

        return ContextGateResult(
            selected_results=selected_results,
            minimum_reranker_score=self._minimum_reranker_score,
        )

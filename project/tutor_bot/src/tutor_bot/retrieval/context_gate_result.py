from pydantic import BaseModel, ConfigDict, computed_field

from tutor_bot.retrieval.hybrid_search_result import HybridSearchResult


class ContextGateResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    selected_results: tuple[HybridSearchResult, ...]
    minimum_reranker_score: float

    @computed_field
    @property
    def has_sufficient_context(self) -> bool:

        return bool(self.selected_results)

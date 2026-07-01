from pydantic import BaseModel, ConfigDict, Field

from tutor_bot.retrieval.chunk_search_result import ChunkSearchResult


class HybridSearchResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    chunk: ChunkSearchResult
    rrf_score: float = Field(gt=0)
    reranker_score: float
    vector_rank: int | None
    bm25_rank: int | None

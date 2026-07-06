from uuid import UUID

from tutor_bot.indexing.bm25_chunk_index import Bm25ChunkIndex
from tutor_bot.indexing.chroma_chunk_index import ChromaChunkIndex
from tutor_bot.indexing.chunk_search_text import build_chunk_search_text
from tutor_bot.indexing.embedding_service import EmbeddingService
from tutor_bot.retrieval.chunk_search_result import ChunkSearchResult
from tutor_bot.retrieval.hybrid_search_result import HybridSearchResult
from tutor_bot.retrieval.reranker import Reranker


class HybridSearchService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_index: ChromaChunkIndex,
        bm25_index: Bm25ChunkIndex,
        reranker: Reranker,
        max_chunks_per_note: int = 2,
        candidate_limit: int = 20,
        rrf_constant: int = 60,
    ) -> None:
        if candidate_limit <= 0 or rrf_constant <= 0 or max_chunks_per_note <= 0:
            raise ValueError("Search limits and RRF constant must be positive")

        self._embedding_service = embedding_service
        self._vector_index = vector_index
        self._bm25_index = bm25_index
        self._reranker = reranker
        self._max_chunks_per_note = max_chunks_per_note
        self._candidate_limit = candidate_limit
        self._rrf_constant = rrf_constant

    def search(
        self,
        query: str,
        limit: int = 10,
        group: str | None = None,
    ) -> list[HybridSearchResult]:
        if limit <= 0:
            raise ValueError("Search result limit must be positive")

        candidate_limit = max(
            limit,
            self._candidate_limit,
        )

        query_embedding = self._embedding_service.embed_query(query)

        vector_results = self._vector_index.search(
            query_embedding,
            limit=candidate_limit,
            group=group,
        )

        bm25_results = self._bm25_index.search(
            query,
            limit=candidate_limit,
            group=group,
        )

        chunks_by_id: dict[UUID, ChunkSearchResult] = {}
        scores_by_id: dict[UUID, float] = {}
        vector_ranks: dict[UUID, int] = {}
        bm25_ranks: dict[UUID, int] = {}

        self._add_ranked_results(
            vector_results,
            chunks_by_id,
            scores_by_id,
            vector_ranks,
        )

        self._add_ranked_results(
            bm25_results,
            chunks_by_id,
            scores_by_id,
            bm25_ranks,
        )

        ranked_chunk_ids = sorted(
            scores_by_id,
            key=lambda chunk_id: (
                -scores_by_id[chunk_id],
                str(chunk_id),
            ),
        )[:candidate_limit]

        reranker_scores = self._reranker.score(
            query,
            [
                build_chunk_search_text(
                    chunks_by_id[chunk_id].note_title,
                    chunks_by_id[chunk_id].heading_title,
                    chunks_by_id[chunk_id].text,
                )
                for chunk_id in ranked_chunk_ids
            ],
        )

        reranker_scores_by_id = dict(
            zip(
                ranked_chunk_ids,
                reranker_scores,
                strict=True,
            )
        )

        reranked_chunk_ids = sorted(
            ranked_chunk_ids,
            key=lambda chunk_id: (
                -reranker_scores_by_id[chunk_id],
                -scores_by_id[chunk_id],
                str(chunk_id),
            ),
        )

        selected_chunk_ids = []
        chunk_counts_by_note: dict[UUID, int] = {}

        for chunk_id in reranked_chunk_ids:
            note_id = chunks_by_id[chunk_id].note_id
            note_chunk_count = chunk_counts_by_note.get(note_id, 0)

            if note_chunk_count >= self._max_chunks_per_note:
                continue

            selected_chunk_ids.append(chunk_id)
            chunk_counts_by_note[note_id] = note_chunk_count + 1

            if len(selected_chunk_ids) == limit:
                break

        return [
            HybridSearchResult(
                chunk=chunks_by_id[chunk_id],
                rrf_score=scores_by_id[chunk_id],
                reranker_score=reranker_scores_by_id[chunk_id],
                vector_rank=vector_ranks.get(chunk_id),
                bm25_rank=bm25_ranks.get(chunk_id),
            )
            for chunk_id in selected_chunk_ids
        ]

    def _add_ranked_results(
        self,
        results: list[ChunkSearchResult],
        chunks_by_id: dict[UUID, ChunkSearchResult],
        scores_by_id: dict[UUID, float],
        ranks_by_id: dict[UUID, int],
    ) -> None:
        for rank, result in enumerate(
            results,
            start=1,
        ):
            chunk_id = result.chunk_id
            chunks_by_id.setdefault(
                chunk_id,
                result,
            )
            ranks_by_id[chunk_id] = rank
            scores_by_id[chunk_id] = scores_by_id.get(chunk_id, 0.0) + 1.0 / (
                self._rrf_constant + rank
            )

from time import perf_counter
from typing import Protocol
from uuid import uuid4

from tutor_bot.application.tutor_answer import TutorAnswer
from tutor_bot.generation.grounded_answer_generator import GroundedAnswerGenerator
from tutor_bot.retrieval.hybrid_search_service import HybridSearchService
from tutor_bot.retrieval.hybrid_search_result import HybridSearchResult
from tutor_bot.retrieval.reranker_context_gate import RerankerContextGate
from tutor_bot.schemas.observability_event import ObservabilityEvent


class _ObservabilityEventRecorder(Protocol):
    def record(
        self,
        event: ObservabilityEvent,
    ) -> None:
        pass


class TutorAnswerService:
    def __init__(
        self,
        search_service: HybridSearchService,
        context_gate: RerankerContextGate,
        answer_generator: GroundedAnswerGenerator,
        search_limit: int = 10,
        observability_event_service: _ObservabilityEventRecorder | None = None,
    ) -> None:
        if search_limit <= 0:
            raise ValueError("Search limit must be positive")

        self._search_service = search_service
        self._context_gate = context_gate
        self._answer_generator = answer_generator
        self._search_limit = search_limit
        self._observability_event_service = observability_event_service

    def answer(
        self,
        question: str,
    ) -> TutorAnswer:
        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError("Question must not be empty")

        trace_id = str(uuid4())
        pipeline_started_at = perf_counter()
        self._record_event(
            ObservabilityEvent(
                scenario="rag_answer",
                event_type="pipeline",
                status="started",
                trace_id=trace_id,
                payload={
                    "question_length": len(normalized_question),
                    "search_limit": self._search_limit,
                },
            )
        )

        try:
            search_started_at = perf_counter()
            search_results = self._search_service.search(
                normalized_question,
                limit=self._search_limit,
            )
            retrieval_duration_seconds = perf_counter() - search_started_at

            context_started_at = perf_counter()
            context = self._context_gate.select(search_results)
            context_duration_seconds = perf_counter() - context_started_at

            generation_started_at = perf_counter()
            generated_answer = self._answer_generator.generate(
                normalized_question,
                context,
            )
            generation_duration_seconds = perf_counter() - generation_started_at

            self._record_event(
                ObservabilityEvent(
                    scenario="rag_answer",
                    event_type="pipeline",
                    status="succeeded",
                    trace_id=trace_id,
                    duration_seconds=perf_counter() - pipeline_started_at,
                    payload={
                        "question_length": len(normalized_question),
                        "search_limit": self._search_limit,
                        "candidate_count": len(search_results),
                        "selected_context_count": len(context.selected_results),
                        "has_sufficient_context": context.has_sufficient_context,
                        "answer_length": len(generated_answer),
                        "retrieval_duration_seconds": round(retrieval_duration_seconds, 3),
                        "context_duration_seconds": round(context_duration_seconds, 3),
                        "generation_duration_seconds": round(generation_duration_seconds, 3),
                        "sources": self._build_source_payload(context.selected_results),
                    },
                )
            )
        except Exception as exception:
            self._record_event(
                ObservabilityEvent(
                    scenario="rag_answer",
                    event_type="pipeline",
                    status="failed",
                    trace_id=trace_id,
                    duration_seconds=perf_counter() - pipeline_started_at,
                    payload={
                        "question_length": len(normalized_question),
                        "search_limit": self._search_limit,
                    },
                    error=exception.__class__.__name__,
                )
            )
            raise

        return TutorAnswer(
            question=normalized_question,
            answer=generated_answer,
            context=context,
        )

    def _record_event(
        self,
        event: ObservabilityEvent,
    ) -> None:
        if self._observability_event_service is None:
            return

        self._observability_event_service.record(event)

    def _build_source_payload(
        self,
        selected_results: tuple[HybridSearchResult, ...],
    ) -> tuple[dict[str, str | int | float | None], ...]:

        return tuple(
            {
                "note_id": str(result.chunk.note_id),
                "chunk_id": str(result.chunk.chunk_id),
                "note_title": result.chunk.note_title,
                "heading_title": result.chunk.heading_title,
                "relative_path": str(result.chunk.relative_path),
                "reranker_score": round(result.reranker_score, 4),
                "rrf_score": round(result.rrf_score, 4),
                "vector_rank": result.vector_rank,
                "bm25_rank": result.bm25_rank,
            }
            for result in selected_results
        )

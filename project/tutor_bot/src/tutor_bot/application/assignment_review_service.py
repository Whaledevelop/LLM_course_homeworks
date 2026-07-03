from time import perf_counter
from typing import Protocol
from uuid import uuid4

from tutor_bot.application.assignment_review_result import AssignmentReviewResult
from tutor_bot.generation.grounded_assignment_reviewer import GroundedAssignmentReviewer
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


class AssignmentReviewService:
    def __init__(
        self,
        search_service: HybridSearchService,
        context_gate: RerankerContextGate,
        reviewer: GroundedAssignmentReviewer,
        search_limit: int = 10,
        observability_event_service: _ObservabilityEventRecorder | None = None,
    ) -> None:
        if search_limit <= 0:
            raise ValueError("Search limit must be positive")

        self._search_service = search_service
        self._context_gate = context_gate
        self._reviewer = reviewer
        self._search_limit = search_limit
        self._observability_event_service = observability_event_service

    def review(
        self,
        assignment_text: str,
        student_answer: str,
    ) -> AssignmentReviewResult:
        normalized_assignment = assignment_text.strip()
        normalized_answer = student_answer.strip()

        if not normalized_assignment or not normalized_answer:
            raise ValueError("Assignment and student answer must not be empty")

        trace_id = str(uuid4())
        pipeline_observation_id = uuid4()
        pipeline_started_at = perf_counter()
        self._record_event(
            ObservabilityEvent(
                scenario="assignment_review",
                event_type="pipeline",
                observation_type="trace",
                observation_id=pipeline_observation_id,
                status="started",
                trace_id=trace_id,
                payload={
                    "assignment_length": len(normalized_assignment),
                    "student_answer_length": len(normalized_answer),
                    "search_limit": self._search_limit,
                },
            )
        )

        try:
            search_started_at = perf_counter()
            search_results = self._search_service.search(
                normalized_assignment,
                limit=self._search_limit,
            )
            retrieval_duration_seconds = perf_counter() - search_started_at
            self._record_event(
                ObservabilityEvent(
                    scenario="assignment_review",
                    event_type="retrieval",
                    observation_type="retriever",
                    parent_observation_id=pipeline_observation_id,
                    status="succeeded",
                    trace_id=trace_id,
                    duration_seconds=retrieval_duration_seconds,
                    payload={"candidate_count": len(search_results)},
                )
            )

            context_started_at = perf_counter()
            context = self._context_gate.select(search_results)
            context_duration_seconds = perf_counter() - context_started_at
            self._record_event(
                ObservabilityEvent(
                    scenario="assignment_review",
                    event_type="reranking_context_gate",
                    observation_type="span",
                    parent_observation_id=pipeline_observation_id,
                    status="succeeded",
                    trace_id=trace_id,
                    duration_seconds=context_duration_seconds,
                    payload={
                        "selected_context_count": len(context.selected_results),
                        "has_sufficient_context": context.has_sufficient_context,
                        "sources": self._build_source_payload(context.selected_results),
                    },
                )
            )

            review_started_at = perf_counter()

            if self._observability_event_service is None:
                review = self._reviewer.review(
                    normalized_assignment,
                    normalized_answer,
                    context,
                )
            else:
                with self._observability_event_service.observe(
                    "assignment_review",
                    "generation",
                    observation_type="generation",
                    payload={
                        "input": {
                            "assignment_length": len(normalized_assignment),
                            "student_answer_length": len(normalized_answer),
                        }
                    },
                    trace_id=trace_id,
                    parent_observation_id=pipeline_observation_id,
                ) as generation_scope:
                    review = self._reviewer.review(
                        normalized_assignment,
                        normalized_answer,
                        context,
                    )
                    generation_scope.set_output(review.model_dump(mode="json"))
                    generation_scope.add_metadata(validation_succeeded=True)

            review_duration_seconds = perf_counter() - review_started_at
            self._record_event(
                ObservabilityEvent(
                    scenario="assignment_review",
                    event_type="generation_validation",
                    observation_type="evaluator",
                    parent_observation_id=pipeline_observation_id,
                    status="succeeded",
                    trace_id=trace_id,
                    duration_seconds=review_duration_seconds,
                    payload={
                        "validation_succeeded": True,
                        "verdict": review.verdict,
                        "output": review.model_dump(mode="json"),
                    },
                )
            )

            self._record_event(
                ObservabilityEvent(
                    scenario="assignment_review",
                    event_type="pipeline",
                    observation_type="trace",
                    observation_id=pipeline_observation_id,
                    status="succeeded",
                    trace_id=trace_id,
                    duration_seconds=perf_counter() - pipeline_started_at,
                    payload={
                        "assignment_length": len(normalized_assignment),
                        "student_answer_length": len(normalized_answer),
                        "search_limit": self._search_limit,
                        "candidate_count": len(search_results),
                        "selected_context_count": len(context.selected_results),
                        "has_sufficient_context": context.has_sufficient_context,
                        "verdict": review.verdict,
                        "correct_points_count": len(review.correct_points),
                        "errors_count": len(review.errors),
                        "missing_points_count": len(review.missing_points),
                        "retrieval_duration_seconds": round(retrieval_duration_seconds, 3),
                        "context_duration_seconds": round(context_duration_seconds, 3),
                        "review_duration_seconds": round(review_duration_seconds, 3),
                        "sources": self._build_source_payload(context.selected_results),
                    },
                )
            )
        except Exception as exception:
            self._record_event(
                ObservabilityEvent(
                    scenario="assignment_review",
                    event_type="pipeline",
                    observation_type="trace",
                    observation_id=pipeline_observation_id,
                    status="failed",
                    trace_id=trace_id,
                    duration_seconds=perf_counter() - pipeline_started_at,
                    payload={
                        "assignment_length": len(normalized_assignment),
                        "student_answer_length": len(normalized_answer),
                        "search_limit": self._search_limit,
                    },
                    error=exception.__class__.__name__,
                )
            )
            raise

        return AssignmentReviewResult(
            assignment_text=normalized_assignment,
            student_answer=normalized_answer,
            review=review,
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

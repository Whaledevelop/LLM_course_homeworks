from tutor_bot.application.assignment_review_result import AssignmentReviewResult
from tutor_bot.generation.grounded_assignment_reviewer import GroundedAssignmentReviewer
from tutor_bot.retrieval.hybrid_search_service import HybridSearchService
from tutor_bot.retrieval.reranker_context_gate import RerankerContextGate


class AssignmentReviewService:
    def __init__(
        self,
        search_service: HybridSearchService,
        context_gate: RerankerContextGate,
        reviewer: GroundedAssignmentReviewer,
        search_limit: int = 10,
    ) -> None:
        if search_limit <= 0:
            raise ValueError("Search limit must be positive")

        self._search_service = search_service
        self._context_gate = context_gate
        self._reviewer = reviewer
        self._search_limit = search_limit

    def review(
        self,
        assignment_text: str,
        student_answer: str,
    ) -> AssignmentReviewResult:
        normalized_assignment = assignment_text.strip()
        normalized_answer = student_answer.strip()

        if not normalized_assignment or not normalized_answer:
            raise ValueError("Assignment and student answer must not be empty")

        search_results = self._search_service.search(
            normalized_assignment,
            limit=self._search_limit,
        )
        context = self._context_gate.select(search_results)
        review = self._reviewer.review(
            normalized_assignment,
            normalized_answer,
            context,
        )

        return AssignmentReviewResult(
            assignment_text=normalized_assignment,
            student_answer=normalized_answer,
            review=review,
            context=context,
        )

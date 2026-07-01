from typing import Protocol

from tutor_bot.application.assignment_review import AssignmentReview
from tutor_bot.retrieval.context_gate_result import ContextGateResult


class GroundedAssignmentReviewer(Protocol):
    def review(
        self,
        assignment_text: str,
        student_answer: str,
        context: ContextGateResult,
    ) -> AssignmentReview: ...

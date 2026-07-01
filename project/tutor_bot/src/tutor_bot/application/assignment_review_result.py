from pydantic import BaseModel, ConfigDict, Field

from tutor_bot.application.assignment_review import AssignmentReview
from tutor_bot.retrieval.context_gate_result import ContextGateResult


class AssignmentReviewResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    assignment_text: str = Field(min_length=1)
    student_answer: str = Field(min_length=1)
    review: AssignmentReview
    context: ContextGateResult

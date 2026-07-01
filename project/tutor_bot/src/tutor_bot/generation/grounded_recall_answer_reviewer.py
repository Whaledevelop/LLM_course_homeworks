from typing import Protocol

from tutor_bot.application.recall_answer_review import RecallAnswerReview
from tutor_bot.application.recall_exercise import RecallExercise


class GroundedRecallAnswerReviewer(Protocol):
    def review(
        self,
        exercise: RecallExercise,
        student_answer: str,
    ) -> RecallAnswerReview: ...

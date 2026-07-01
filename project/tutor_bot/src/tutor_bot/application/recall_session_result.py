from pydantic import BaseModel, ConfigDict, Field

from tutor_bot.application.recall_answer_review import RecallAnswerReview
from tutor_bot.application.recall_session import RecallSession


class RecallSessionResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    session: RecallSession
    student_answer: str = Field(min_length=1)
    review: RecallAnswerReview

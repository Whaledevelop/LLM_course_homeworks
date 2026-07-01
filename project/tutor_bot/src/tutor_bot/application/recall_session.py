from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from tutor_bot.application.recall_exercise import RecallExercise


class RecallSession(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    note_id: UUID
    note_title: str = Field(min_length=1)
    exercise: RecallExercise

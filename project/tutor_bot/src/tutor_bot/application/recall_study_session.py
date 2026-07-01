from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from tutor_bot.application.recall_session import RecallSession
from tutor_bot.application.recall_session_result import RecallSessionResult


class RecallStudySession(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    seed: int
    note_ids: tuple[UUID, ...] = Field(
        min_length=1,
        max_length=10,
    )
    current_index: int = Field(
        ge=0,
        lt=10,
    )
    current_exercise: RecallSession
    results: tuple[RecallSessionResult, ...] = ()

    @property
    def answered_count(self) -> int:
        return len(self.results)

    @property
    def total_count(self) -> int:
        return len(self.note_ids)

    @property
    def is_complete(self) -> bool:
        return self.answered_count == self.total_count

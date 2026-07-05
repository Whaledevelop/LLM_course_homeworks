from pydantic import BaseModel, ConfigDict, Field

from tutor_bot.application.recall_session import RecallSession
from tutor_bot.application.recall_session_result import RecallSessionResult
from tutor_bot.application.vacancy_study_target import VacancyStudyTarget


class VacancyStudySession(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    vacancy_title: str = Field(min_length=1)
    targets: tuple[VacancyStudyTarget, ...] = Field(min_length=1)
    current_index: int = Field(ge=0)
    current_exercise: RecallSession
    results: tuple[RecallSessionResult, ...] = ()
    reviewed_indices: tuple[int, ...] = ()
    imitated_indices: tuple[int, ...] = ()

    @property
    def answered_count(self) -> int:
        return len(self.reviewed_indices) + len(self.imitated_indices)

    @property
    def total_count(self) -> int:
        return len(self.targets)

    @property
    def is_complete(self) -> bool:
        return self.answered_count == self.total_count

    @property
    def current_question_is_reviewed(self) -> bool:
        return self.current_index in self.reviewed_indices

    @property
    def current_question_is_imitated(self) -> bool:
        return self.current_index in self.imitated_indices

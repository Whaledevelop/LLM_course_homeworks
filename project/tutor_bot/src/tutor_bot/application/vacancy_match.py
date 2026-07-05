from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from tutor_bot.application.vacancy_requirement import VacancyRequirement


class VacancyMatch(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    requirement: VacancyRequirement
    note_id: UUID | None = None
    note_title: str | None = None
    confidence: float = Field(ge=0, le=1)

    @property
    def is_covered(self) -> bool:
        return self.note_id is not None

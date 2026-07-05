from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from tutor_bot.application.vacancy_requirement import VacancyRequirement


class Vacancy(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    id: UUID
    sha256: str = Field(min_length=64, max_length=64)
    original_filename: str = Field(min_length=1)
    title: str = Field(min_length=1)
    uploaded_at: datetime
    extracted_text: str = Field(min_length=1)
    requirements: tuple[VacancyRequirement, ...] = Field(min_length=1)

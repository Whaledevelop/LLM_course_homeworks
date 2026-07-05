from pydantic import BaseModel, ConfigDict, Field

from tutor_bot.application.vacancy_requirement import VacancyRequirement


class VacancyAnalysis(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    title: str = Field(min_length=1)
    requirements: tuple[VacancyRequirement, ...] = Field(min_length=1)

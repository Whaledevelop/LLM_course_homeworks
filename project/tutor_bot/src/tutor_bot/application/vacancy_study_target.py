from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class VacancyStudyTarget(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    requirement_id: UUID
    topic: str = Field(min_length=1)
    expected_knowledge: str = Field(min_length=1)
    note_id: UUID
    note_title: str = Field(min_length=1)

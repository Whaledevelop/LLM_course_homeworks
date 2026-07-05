from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class VacancyRequirement(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    id: UUID = Field(default_factory=uuid4)
    topic: str = Field(min_length=1)
    expected_knowledge: str = Field(min_length=1)
    evidence: str = Field(min_length=1)

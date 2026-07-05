from pydantic import BaseModel, ConfigDict, Field


class VacancyMatchDecision(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    matched_note_title: str | None = None
    confidence: float = Field(ge=0, le=1)

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EvaluationCorpus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    note_ids: list[UUID] = Field(min_length=15, max_length=20)

    @field_validator("note_ids")
    @classmethod
    def validate_unique_note_ids(
        cls,
        note_ids: list[UUID],
    ) -> list[UUID]:
        if len(note_ids) != len(set(note_ids)):
            raise ValueError("Evaluation note ids must be unique")

        return note_ids

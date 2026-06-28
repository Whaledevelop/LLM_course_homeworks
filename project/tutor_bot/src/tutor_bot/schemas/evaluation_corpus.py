from pathlib import Path
from typing import Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvaluationCorpus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    notes: dict[UUID, Path] = Field(min_length=15, max_length=20)

    @model_validator(mode="after")
    def validate_note_paths(self) -> Self:
        note_paths = list(self.notes.values())

        if len(note_paths) != len(set(note_paths)):
            raise ValueError("Evaluation note paths must be unique")

        for note_path in note_paths:
            if note_path.is_absolute() or note_path.suffix.lower() != ".md":
                raise ValueError("Evaluation notes must be relative Markdown paths")

        return self
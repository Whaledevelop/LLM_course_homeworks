from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NoteMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    theme: str
    comment: str
    difficulty: str
    importance: int = Field(ge=0, le=10)
    completeness: int = Field(ge=0, le=10)
    mastery: int = Field(ge=0, le=10)
    last_recorded_name: str = Field(min_length=1)
    relative_path: Path

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, relative_path: Path) -> Path:
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.suffix.lower() != ".md"
        ):
            raise ValueError("Note path must be a relative Markdown path")

        return relative_path

from datetime import datetime
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NoteMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group: str
    comment: str
    questions_for_tests: tuple[str, ...] = ()
    importance: int = Field(ge=0, le=10)
    knowledge: int = Field(ge=0, le=10)
    fullness: int = Field(default=0, ge=0, le=10)
    time_added: datetime = Field(default_factory=lambda: datetime.now().astimezone())
    last_recorded_name: str = Field(min_length=1)
    relative_path: PurePosixPath

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(
        cls,
        relative_path: PurePosixPath,
    ) -> PurePosixPath:
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.suffix.lower() != ".md"
        ):
            raise ValueError("Note path must be a relative Markdown path")

        return relative_path

import re
from pathlib import Path, PurePosixPath
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


_DB_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class DatabaseRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root_path: Path


class DatabaseRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_db_id: str | None = None
    databases: dict[str, DatabaseRegistration] = Field(default_factory=dict)

    @field_validator("active_db_id")
    @classmethod
    def validate_active_db_id(cls, db_id: str | None) -> str | None:
        if db_id is not None:
            validate_db_id(db_id)

        return db_id

    @field_validator("databases")
    @classmethod
    def validate_database_ids(
        cls,
        databases: dict[str, DatabaseRegistration],
    ) -> dict[str, DatabaseRegistration]:
        for db_id in databases:
            validate_db_id(db_id)

        return databases


class DatabaseIndexNote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: PurePosixPath

    @field_validator("path")
    @classmethod
    def validate_path(cls, path: PurePosixPath) -> PurePosixPath:
        if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".md":
            raise ValueError("Note path must be a relative Markdown path")

        return path


class DatabaseIndex(BaseModel):
    model_config = ConfigDict(extra="forbid")

    db_id: str
    notes: dict[UUID, DatabaseIndexNote] = Field(default_factory=dict)

    @field_validator("db_id")
    @classmethod
    def validate_identifier(cls, db_id: str) -> str:
        return validate_db_id(db_id)


class DatabaseNoteMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    group: str = ""
    comment: str = ""
    questions_for_tests: list[str] = Field(default_factory=list)
    importance: int = Field(default=0, ge=0, le=10)
    knowledge: int = Field(default=0, ge=0, le=10)
    fullness: int | None = Field(default=None, ge=0, le=10)


class DatabaseMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    db_id: str
    notes: dict[UUID, DatabaseNoteMetadata] = Field(default_factory=dict)

    @field_validator("db_id")
    @classmethod
    def validate_identifier(cls, db_id: str) -> str:
        return validate_db_id(db_id)


def validate_db_id(db_id: str) -> str:
    if not _DB_ID_PATTERN.fullmatch(db_id):
        raise ValueError("DB id may contain only letters, numbers, underscores, and hyphens")

    return db_id

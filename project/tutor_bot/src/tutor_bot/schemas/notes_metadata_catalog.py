from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from tutor_bot.schemas.note_metadata import NoteMetadata


class NotesMetadataCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=2)
    version_time: AwareDatetime
    notes: dict[UUID, NoteMetadata]

    @field_validator("notes")
    @classmethod
    def validate_unique_paths(
        cls,
        notes: dict[UUID, NoteMetadata],
    ) -> dict[UUID, NoteMetadata]:
        normalized_paths = [
            metadata.relative_path.as_posix().casefold() for metadata in notes.values()
        ]

        if len(normalized_paths) != len(set(normalized_paths)):
            raise ValueError("Metadata note paths must be unique")

        return notes

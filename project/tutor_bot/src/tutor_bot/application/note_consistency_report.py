from pathlib import PurePosixPath
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class NoteConsistencyReport(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    missing_markdown_note_ids: tuple[UUID, ...]
    markdown_without_metadata: tuple[PurePosixPath, ...]
    duplicate_markdown_note_ids: tuple[UUID, ...]
    duplicate_metadata_paths: tuple[PurePosixPath, ...]
    frontmatter_id_mismatches: tuple[PurePosixPath, ...]
    invalid_markdown_paths: tuple[PurePosixPath, ...]

    @property
    def is_consistent(self) -> bool:
        issues = (
            self.missing_markdown_note_ids,
            self.markdown_without_metadata,
            self.duplicate_markdown_note_ids,
            self.duplicate_metadata_paths,
            self.frontmatter_id_mismatches,
            self.invalid_markdown_paths,
        )

        return not any(issues)

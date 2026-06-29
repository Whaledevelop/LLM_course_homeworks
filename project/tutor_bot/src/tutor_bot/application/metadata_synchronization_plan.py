from pathlib import PurePosixPath
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from tutor_bot.application.note_consistency_report import (
    NoteConsistencyReport,
)


class MetadataSynchronizationPlan(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    metadata_note_ids_to_remove: tuple[UUID, ...]
    markdown_paths_to_register: tuple[PurePosixPath, ...]
    consistency_report: NoteConsistencyReport

    @property
    def has_changes(self) -> bool:
        changes = (
            self.metadata_note_ids_to_remove,
            self.markdown_paths_to_register,
        )

        return any(changes)

    @property
    def can_apply(self) -> bool:
        blocking_issues = (
            self.consistency_report.duplicate_markdown_note_ids,
            self.consistency_report.duplicate_metadata_paths,
            self.consistency_report.frontmatter_id_mismatches,
            self.consistency_report.invalid_markdown_paths,
        )

        return not any(blocking_issues)

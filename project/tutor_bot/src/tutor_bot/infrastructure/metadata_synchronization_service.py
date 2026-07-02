from pathlib import Path, PurePosixPath

from tutor_bot.infrastructure.markdown_document import MarkdownDocument
from tutor_bot.infrastructure.metadata_synchronization_planner import (
    MetadataSynchronizationPlanner,
)
from tutor_bot.infrastructure.notes_metadata_repository import (
    NotesMetadataRepository,
)
from tutor_bot.schemas.note_metadata import NoteMetadata
from tutor_bot.schemas.notes_metadata_catalog import NotesMetadataCatalog


class MetadataSynchronizationService:
    def __init__(
        self,
        metadata_repository: NotesMetadataRepository,
        synchronization_planner: MetadataSynchronizationPlanner,
        source_notes_dir: Path,
    ) -> None:
        self._metadata_repository = metadata_repository
        self._synchronization_planner = synchronization_planner
        self._source_notes_dir = source_notes_dir.resolve()

    def synchronize(self) -> NotesMetadataCatalog:
        plan = self._synchronization_planner.create_plan()

        if not plan.can_apply:
            raise RuntimeError("Metadata synchronization is blocked by consistency issues")

        catalog = self._metadata_repository.load()
        updated_notes = dict(catalog.notes)

        for note_id in plan.metadata_note_ids_to_remove:
            updated_notes.pop(note_id, None)

        for relative_path in plan.markdown_paths_to_register:
            document = self._read_document(relative_path)

            if document.note_id in updated_notes:
                raise ValueError(f"Metadata already exists for note: {document.note_id}")

            updated_notes[document.note_id] = NoteMetadata(
                theme="",
                comment="",
                importance=0,
                knowledge=0,
                last_recorded_name=relative_path.stem,
                relative_path=relative_path,
            )

        if updated_notes == catalog.notes:
            return catalog

        updated_catalog = catalog.model_copy(update={"notes": updated_notes})

        return self._metadata_repository.save(updated_catalog)

    def _read_document(
        self,
        relative_path: PurePosixPath,
    ) -> MarkdownDocument:
        note_path = (self._source_notes_dir / relative_path).resolve()

        if not note_path.is_relative_to(self._source_notes_dir):
            raise ValueError(f"Note path escapes source directory: {note_path}")

        file_content = note_path.read_text(encoding="utf-8-sig")

        return MarkdownDocument.parse(
            file_content,
            note_path,
        )

from pathlib import Path
from uuid import UUID, uuid4

from tutor_bot.application.create_note_command import CreateNoteCommand
from tutor_bot.application.delete_note_command import DeleteNoteCommand
from tutor_bot.application.note_details import NoteDetails
from tutor_bot.application.update_note_command import UpdateNoteCommand
from tutor_bot.infrastructure.markdown_document import MarkdownDocument
from tutor_bot.infrastructure.markdown_note_storage import (
    MarkdownNoteStorage,
)
from tutor_bot.infrastructure.notes_metadata_repository import (
    NotesMetadataRepository,
)
from tutor_bot.schemas.note_metadata import NoteMetadata
from tutor_bot.schemas.notes_metadata_catalog import NotesMetadataCatalog


class FileNoteCommandService:
    def __init__(
        self,
        metadata_repository: NotesMetadataRepository,
        source_notes_dir: Path,
    ) -> None:
        self._metadata_repository = metadata_repository
        self._markdown_storage = MarkdownNoteStorage(source_notes_dir)

    def create_note(
        self,
        command: CreateNoteCommand,
    ) -> NoteDetails:
        catalog = self._metadata_repository.load()
        note_id = uuid4()

        relative_path, document = self._markdown_storage.create(
            note_id,
            command.markdown_content,
        )

        metadata = NoteMetadata(
            theme=command.theme,
            comment=command.comment,
            difficulty=command.difficulty,
            importance=command.importance,
            completeness=command.completeness,
            mastery=command.mastery,
            last_recorded_name=command.title,
            relative_path=relative_path,
        )

        updated_catalog = self._replace_metadata(
            catalog,
            note_id,
            metadata,
        )

        try:
            self._metadata_repository.save(updated_catalog)
        except Exception:
            self._markdown_storage.delete_created(relative_path)

            raise

        return self._create_details(
            note_id,
            metadata,
            document,
        )

    def update_note(
        self,
        command: UpdateNoteCommand,
    ) -> NoteDetails:
        catalog = self._metadata_repository.load()
        metadata = self._get_metadata(
            catalog,
            command.note_id,
        )

        original_file_content, document = self._markdown_storage.update(
            command.note_id,
            metadata.relative_path,
            command.markdown_content,
        )

        updated_metadata = metadata.model_copy(
            update={
                "last_recorded_name": command.title,
                "theme": command.theme,
                "comment": command.comment,
                "difficulty": command.difficulty,
                "importance": command.importance,
                "completeness": command.completeness,
                "mastery": command.mastery,
            }
        )

        updated_catalog = self._replace_metadata(
            catalog,
            command.note_id,
            updated_metadata,
        )

        try:
            self._metadata_repository.save(updated_catalog)
        except Exception:
            self._markdown_storage.restore(
                metadata.relative_path,
                original_file_content,
            )

            raise

        return self._create_details(
            command.note_id,
            updated_metadata,
            document,
        )

    def delete_note(
        self,
        command: DeleteNoteCommand,
    ) -> NoteDetails:
        catalog = self._metadata_repository.load()
        metadata = self._get_metadata(
            catalog,
            command.note_id,
        )

        staged_file, document = self._markdown_storage.stage_delete(
            command.note_id,
            metadata.relative_path,
        )

        updated_notes = dict(catalog.notes)
        del updated_notes[command.note_id]
        updated_catalog = catalog.model_copy(update={"notes": updated_notes})

        try:
            self._metadata_repository.save(updated_catalog)
        except Exception:
            self._markdown_storage.restore_deleted(
                staged_file,
                metadata.relative_path,
            )

            raise

        self._markdown_storage.finalize_deleted(staged_file)

        return self._create_details(
            command.note_id,
            metadata,
            document,
        )

    def _get_metadata(
        self,
        catalog: NotesMetadataCatalog,
        note_id: UUID,
    ) -> NoteMetadata:
        if note_id not in catalog.notes:
            raise KeyError(f"Note metadata not found: {note_id}")

        return catalog.notes[note_id]

    def _replace_metadata(
        self,
        catalog: NotesMetadataCatalog,
        note_id: UUID,
        metadata: NoteMetadata,
    ) -> NotesMetadataCatalog:
        updated_notes = dict(catalog.notes)
        updated_notes[note_id] = metadata

        return catalog.model_copy(update={"notes": updated_notes})

    def _create_details(
        self,
        note_id: UUID,
        metadata: NoteMetadata,
        document: MarkdownDocument,
    ) -> NoteDetails:
        return NoteDetails(
            id=note_id,
            title=metadata.last_recorded_name,
            theme=metadata.theme,
            difficulty=metadata.difficulty,
            importance=metadata.importance,
            completeness=metadata.completeness,
            mastery=metadata.mastery,
            comment=metadata.comment,
            markdown_content=document.normalized_content,
        )

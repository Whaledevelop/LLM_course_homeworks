import os
import re
from pathlib import Path
from uuid import UUID, uuid4

from tutor_bot.application.note_details import NoteDetails
from tutor_bot.application.update_note_command import UpdateNoteCommand
from tutor_bot.infrastructure.notes_metadata_repository import (
    NotesMetadataRepository,
)


_FRONTMATTER_PATTERN = re.compile(
    r"\A---\s*\n(?P<content>.*?)\n---(?:\n|\Z)",
    re.DOTALL,
)

_NOTE_ID_PATTERN = re.compile(
    r"^id:\s*(?P<note_id>[^\s]+)\s*$",
    re.MULTILINE,
)


class FileNoteCommandService:
    def __init__(
        self,
        metadata_repository: NotesMetadataRepository,
        source_notes_dir: Path,
    ) -> None:
        self._metadata_repository = metadata_repository
        self._source_notes_dir = source_notes_dir

    def update_note(
        self,
        command: UpdateNoteCommand,
    ) -> NoteDetails:
        catalog = self._metadata_repository.load()

        if command.note_id not in catalog.notes:
            raise KeyError(f"Note metadata not found: {command.note_id}")

        metadata = catalog.notes[command.note_id]
        source_root = self._source_notes_dir.resolve()
        note_path = (source_root / metadata.relative_path).resolve()

        if not note_path.is_relative_to(source_root):
            raise ValueError(f"Note path escapes source directory: {note_path}")

        if not note_path.is_file():
            raise FileNotFoundError(f"Note file not found: {note_path}")

        original_content = note_path.read_text(encoding="utf-8-sig")
        self._validate_note_id(
            original_content,
            command.note_id,
            note_path,
        )

        saved_markdown_content = command.markdown_content.rstrip() + "\n"
        updated_file_content = f"---\nid: {command.note_id}\n---\n\n{saved_markdown_content}"

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

        updated_notes = dict(catalog.notes)
        updated_notes[command.note_id] = updated_metadata
        updated_catalog = catalog.model_copy(update={"notes": updated_notes})

        self._write_atomically(
            note_path,
            updated_file_content,
        )

        try:
            self._metadata_repository.save(updated_catalog)
        except Exception:
            self._write_atomically(
                note_path,
                original_content,
            )

            raise

        return NoteDetails(
            id=command.note_id,
            title=command.title,
            theme=command.theme,
            difficulty=command.difficulty,
            importance=command.importance,
            completeness=command.completeness,
            mastery=command.mastery,
            comment=command.comment,
            markdown_content=saved_markdown_content,
        )

    def _validate_note_id(
        self,
        content: str,
        expected_note_id: UUID,
        note_path: Path,
    ) -> None:
        frontmatter_match = _FRONTMATTER_PATTERN.match(content)

        if frontmatter_match is None:
            raise ValueError(f"Markdown frontmatter not found: {note_path}")

        note_id_match = _NOTE_ID_PATTERN.search(frontmatter_match.group("content"))

        if note_id_match is None:
            raise ValueError(f"Note id not found in frontmatter: {note_path}")

        actual_note_id = UUID(note_id_match.group("note_id"))

        if actual_note_id != expected_note_id:
            raise ValueError(
                f"Note id mismatch: expected {expected_note_id}, found {actual_note_id}"
            )

    def _write_atomically(
        self,
        path: Path,
        content: str,
    ) -> None:
        temporary_file = path.with_name(f".{path.name}.{uuid4().hex}.tmp")

        try:
            with temporary_file.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())

            temporary_file.replace(path)
        finally:
            temporary_file.unlink(missing_ok=True)

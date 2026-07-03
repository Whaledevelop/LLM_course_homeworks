import re
from pathlib import Path
from uuid import UUID

from tutor_bot.application.note_details import NoteDetails
from tutor_bot.application.note_list_item import NoteListItem
from tutor_bot.infrastructure.notes_metadata_repository import (
    NotesMetadataRepository,
)


_FRONTMATTER_PATTERN = re.compile(
    r"\A---\s*\n(?P<content>.*?)\n---(?:\n|\Z)",
    re.DOTALL,
)

_NOTE_ID_PATTERN = re.compile(
    r"^tutor_bot_note_id:\s*(?P<note_id>[^\s]+)\s*$",
    re.MULTILINE,
)


class MetadataNoteQueryService:
    def __init__(
        self,
        metadata_repository: NotesMetadataRepository,
        source_notes_dir: Path,
    ) -> None:
        self._metadata_repository = metadata_repository
        self._source_notes_dir = source_notes_dir

    def list_notes(self) -> list[NoteListItem]:
        metadata_catalog = self._metadata_repository.load()

        note_items = [
            NoteListItem(
                id=note_id,
                title=metadata.last_recorded_name,
                group=metadata.group,
                importance=metadata.importance,
                knowledge=metadata.knowledge,
            )
            for note_id, metadata in metadata_catalog.notes.items()
        ]

        return sorted(
            note_items,
            key=lambda note_item: note_item.title.casefold(),
        )

    def get_note(self, note_id: UUID) -> NoteDetails:
        metadata_catalog = self._metadata_repository.load()

        if note_id not in metadata_catalog.notes:
            raise KeyError(f"Note metadata not found: {note_id}")

        metadata = metadata_catalog.notes[note_id]
        source_root = self._source_notes_dir.resolve()
        note_path = (source_root / metadata.relative_path).resolve()

        if not note_path.is_relative_to(source_root):
            raise ValueError(f"Note path escapes source directory: {note_path}")

        if not note_path.is_file():
            raise FileNotFoundError(f"Note file not found: {note_path}")

        content = note_path.read_text(encoding="utf-8-sig")
        frontmatter_match = _FRONTMATTER_PATTERN.match(content)

        if frontmatter_match is None:
            raise ValueError(f"Markdown frontmatter not found: {note_path}")

        note_id_match = _NOTE_ID_PATTERN.search(frontmatter_match.group("content"))

        if note_id_match is None:
            raise ValueError(f"Note id not found in frontmatter: {note_path}")

        actual_note_id = UUID(note_id_match.group("note_id"))

        if actual_note_id != note_id:
            raise ValueError(f"Note id mismatch: expected {note_id}, found {actual_note_id}")

        markdown_content = content[frontmatter_match.end() :].lstrip("\r\n")

        return NoteDetails(
            id=note_id,
            title=metadata.last_recorded_name,
            group=metadata.group,
            importance=metadata.importance,
            knowledge=metadata.knowledge,
            comment=metadata.comment,
            markdown_content=markdown_content,
        )

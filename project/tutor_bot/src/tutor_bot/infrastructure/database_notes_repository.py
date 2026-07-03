import re
from datetime import datetime
from pathlib import Path

from tutor_bot.infrastructure.database_repository import DatabaseRepository
from tutor_bot.schemas.database import (
    DatabaseIndex,
    DatabaseIndexNote,
    DatabaseMetadata,
    DatabaseNoteMetadata,
)
from tutor_bot.schemas.note_metadata import NoteMetadata
from tutor_bot.schemas.notes_metadata_catalog import NotesMetadataCatalog


_TITLE_PATTERN = re.compile(r"^#\s+(?P<title>.+?)\s*$", re.MULTILINE)
_FRONTMATTER_TITLE_PATTERN = re.compile(r"^title:\s*(?P<title>.+?)\s*$", re.MULTILINE)


class DatabaseNotesRepository:
    def __init__(self, metadata_dir: Path, db_id: str, root_path: Path) -> None:
        self._repository = DatabaseRepository(metadata_dir)
        self._db_id = db_id
        self._root_path = root_path

    def load(self) -> NotesMetadataCatalog:
        index = self._repository.load_index(self._root_path, self._db_id)
        metadata = self._repository.load_metadata(self._db_id)
        notes = {
            note_id: NoteMetadata(
                group=note_metadata.group,
                comment=note_metadata.comment,
                importance=note_metadata.importance,
                knowledge=note_metadata.knowledge,
                last_recorded_name=self._load_title(index_note.path),
                relative_path=index_note.path,
            )
            for note_id, index_note in index.notes.items()
            if (note_metadata := metadata.notes.get(note_id)) is not None
        }

        return NotesMetadataCatalog(
            version=2,
            version_time=datetime.now().astimezone(),
            notes=notes,
        )

    def save(self, catalog: NotesMetadataCatalog) -> NotesMetadataCatalog:
        current_index = self._repository.load_index(self._root_path, self._db_id)
        index_notes = {
            note_id: current_index.notes[note_id].model_copy(
                update={"path": note_metadata.relative_path}
            )
            if note_id in current_index.notes
            else DatabaseIndexNote(path=note_metadata.relative_path)
            for note_id, note_metadata in catalog.notes.items()
        }
        metadata_notes = {
            note_id: DatabaseNoteMetadata(
                group=note_metadata.group,
                comment=note_metadata.comment,
                importance=note_metadata.importance,
                knowledge=note_metadata.knowledge,
            )
            for note_id, note_metadata in catalog.notes.items()
        }
        self._repository.save_index(
            self._root_path,
            DatabaseIndex(db_id=self._db_id, notes=index_notes),
        )
        self._repository.save_metadata(
            DatabaseMetadata(db_id=self._db_id, notes=metadata_notes),
        )

        return catalog

    def _load_title(self, relative_path) -> str:
        note_path = self._root_path / relative_path
        content = note_path.read_text(encoding="utf-8-sig")
        frontmatter_title_match = _FRONTMATTER_TITLE_PATTERN.search(content)

        if frontmatter_title_match is not None:
            return frontmatter_title_match.group("title")

        title_match = _TITLE_PATTERN.search(content)

        if title_match is not None:
            return title_match.group("title")

        return note_path.stem

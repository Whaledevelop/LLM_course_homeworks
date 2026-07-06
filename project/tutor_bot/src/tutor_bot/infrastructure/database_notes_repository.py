from datetime import datetime
from pathlib import Path

from tutor_bot.infrastructure.database_repository import DatabaseRepository
from tutor_bot.application.note_fullness import estimate_note_fullness
from tutor_bot.schemas.database import (
    DatabaseIndex,
    DatabaseIndexNote,
    DatabaseMetadata,
    DatabaseNoteMetadata,
)
from tutor_bot.schemas.note_metadata import NoteMetadata
from tutor_bot.schemas.notes_metadata_catalog import NotesMetadataCatalog


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
                questions_for_tests=tuple(note_metadata.questions_for_tests),
                importance=note_metadata.importance,
                knowledge=note_metadata.knowledge,
                fullness=(
                    note_metadata.fullness
                    if note_metadata.fullness is not None
                    else self._load_fullness(index_note.path)
                ),
                time_added=note_metadata.time_added,
                favorite=note_metadata.favorite,
                last_recorded_name=note_metadata.title or self._load_title(index_note.path),
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
                title=note_metadata.last_recorded_name,
                group=note_metadata.group,
                comment=note_metadata.comment,
                questions_for_tests=list(note_metadata.questions_for_tests),
                importance=note_metadata.importance,
                knowledge=note_metadata.knowledge,
                fullness=note_metadata.fullness,
                favorite=note_metadata.favorite,
                time_added=note_metadata.time_added,
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
        return relative_path.stem

    def _load_fullness(self, relative_path) -> int:
        note_path = self._root_path / relative_path

        return estimate_note_fullness(note_path.read_text(encoding="utf-8-sig"))

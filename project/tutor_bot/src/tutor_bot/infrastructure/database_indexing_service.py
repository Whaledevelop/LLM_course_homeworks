from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from tutor_bot.infrastructure.database_repository import DatabaseRepository
from tutor_bot.infrastructure.note_frontmatter import (
    PreparedNote,
    prepare_note,
    write_note_atomically,
)
from tutor_bot.schemas.database import (
    DatabaseIndex,
    DatabaseIndexNote,
    DatabaseMetadata,
    DatabaseNoteMetadata,
)


@dataclass(frozen=True, slots=True)
class DatabaseIndexingResult:
    added: int
    moved: int
    unchanged: int
    archived: int
    restored: int


class DatabaseIndexingService:
    def __init__(self, repository: DatabaseRepository) -> None:
        self._repository = repository

    def update(self, db_id: str, root_path: Path) -> DatabaseIndexingResult:
        root_path = root_path.resolve()

        if not root_path.is_dir():
            raise NotADirectoryError(f"Notes directory not found: {root_path}")

        old_index = self._repository.load_index(root_path, db_id)

        if old_index.db_id != db_id:
            raise ValueError(
                f"Directory is already indexed as DB '{old_index.db_id}', not '{db_id}'"
            )

        prepared_notes = self._prepare_notes(root_path)
        new_index = self._create_index(db_id, root_path, prepared_notes)
        metadata = self._repository.load_metadata(db_id)
        archive = self._repository.load_archive(db_id)
        updated_metadata, updated_archive, restored = self._synchronize_metadata(
            db_id,
            metadata,
            archive,
            new_index,
        )
        originals = {path: note.original_content for path, note in prepared_notes.items()}

        try:
            for path, note in prepared_notes.items():
                if note.changed:
                    write_note_atomically(path, note.updated_content)

            self._repository.save_index(root_path, new_index)
            self._repository.save_metadata(updated_metadata)
            self._repository.save_archive(updated_archive)
        except Exception:
            for path, original_content in originals.items():
                if path.is_file() and path.read_text(encoding="utf-8-sig") != original_content:
                    write_note_atomically(path, original_content)

            raise

        old_paths = {note_id: note.path for note_id, note in old_index.notes.items()}
        new_paths = {note_id: note.path for note_id, note in new_index.notes.items()}
        added = len(new_paths.keys() - old_paths.keys())
        moved = sum(
            old_paths[note_id] != new_paths[note_id]
            for note_id in new_paths.keys() & old_paths.keys()
        )
        archived = len(old_paths.keys() - new_paths.keys())
        unchanged = len(new_paths) - added - moved

        return DatabaseIndexingResult(added, moved, unchanged, archived, restored)

    def _prepare_notes(self, root_path: Path) -> dict[Path, PreparedNote]:
        prepared_notes: dict[Path, PreparedNote] = {}
        note_paths = sorted(
            path
            for path in root_path.rglob("*.md")
            if not any(part.startswith(".") for part in path.relative_to(root_path).parts)
        )

        for path in note_paths:
            prepared_notes[path] = prepare_note(path)

        note_ids = [note.note_id for note in prepared_notes.values()]

        if len(note_ids) != len(set(note_ids)):
            raise ValueError("Duplicate tutor_bot_note_id found in notes directory")

        return prepared_notes

    def _create_index(
        self,
        db_id: str,
        root_path: Path,
        prepared_notes: dict[Path, PreparedNote],
    ) -> DatabaseIndex:
        notes = {
            note.note_id: DatabaseIndexNote(
                path=path.relative_to(root_path).as_posix(),
            )
            for path, note in prepared_notes.items()
        }

        return DatabaseIndex(db_id=db_id, notes=notes)

    def _synchronize_metadata(
        self,
        db_id: str,
        metadata: DatabaseMetadata,
        archive: DatabaseMetadata,
        index: DatabaseIndex,
    ) -> tuple[DatabaseMetadata, DatabaseMetadata, int]:
        active_notes: dict[UUID, DatabaseNoteMetadata] = {}
        archived_notes = dict(archive.notes)
        restored = 0

        for note_id in index.notes:
            if note_id in metadata.notes:
                active_notes[note_id] = metadata.notes[note_id]
            elif note_id in archived_notes:
                active_notes[note_id] = archived_notes.pop(note_id)
                restored += 1
            else:
                active_notes[note_id] = DatabaseNoteMetadata()

        for note_id, note_metadata in metadata.notes.items():
            if note_id not in index.notes:
                archived_notes[note_id] = note_metadata

        return (
            DatabaseMetadata(db_id=db_id, notes=active_notes),
            DatabaseMetadata(db_id=db_id, notes=archived_notes),
            restored,
        )

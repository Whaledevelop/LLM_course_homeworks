from pathlib import Path
from uuid import UUID

from tutor_bot.infrastructure.database_notes_repository import DatabaseNotesRepository
from tutor_bot.infrastructure.database_repository import DatabaseRepository
from tutor_bot.schemas.database import (
    DatabaseIndex,
    DatabaseIndexNote,
    DatabaseMetadata,
    DatabaseNoteMetadata,
)


def test_uses_current_path_stem_as_note_title(tmp_path: Path) -> None:
    note_id = UUID("0b2c61a8-505a-552c-b731-7b9627970eff")
    notes_root = tmp_path / "notes"
    metadata_dir = tmp_path / "metadata"
    note_path = notes_root / "nested" / "Actual note name.md"
    note_path.parent.mkdir(parents=True)
    note_path.write_text(
        f"---\ntutor_bot_note_id: {note_id}\ntitle: Outdated title\n---\n# Heading\n"
    )
    repository = DatabaseRepository(metadata_dir)
    repository.save_index(
        notes_root,
        DatabaseIndex(
            db_id="Unity",
            notes={
                note_id: DatabaseIndexNote(path="nested/Actual note name.md"),
            },
        ),
    )
    repository.save_metadata(
        DatabaseMetadata(
            db_id="Unity",
            notes={
                note_id: DatabaseNoteMetadata(),
            },
        )
    )

    catalog = DatabaseNotesRepository(metadata_dir, "Unity", notes_root).load()

    assert catalog.notes[note_id].last_recorded_name == "Actual note name"
    assert catalog.notes[note_id].relative_path.as_posix() == "nested/Actual note name.md"
    assert catalog.notes[note_id].fullness == 1


def test_persists_explicit_note_title(tmp_path: Path) -> None:
    note_id = UUID("0b2c61a8-505a-552c-b731-7b9627970eff")
    notes_root = tmp_path / "notes"
    metadata_dir = tmp_path / "metadata"
    note_path = notes_root / "_tutor_bot" / f"{note_id}.md"
    note_path.parent.mkdir(parents=True)
    note_path.write_text(f"---\ntutor_bot_note_id: {note_id}\n---\nContent\n")
    repository = DatabaseRepository(metadata_dir)
    repository.save_index(
        notes_root,
        DatabaseIndex(
            db_id="Unity",
            notes={note_id: DatabaseIndexNote(path=f"_tutor_bot/{note_id}.md")},
        ),
    )
    repository.save_metadata(
        DatabaseMetadata(
            db_id="Unity",
            notes={note_id: DatabaseNoteMetadata(title="Readable title")},
        )
    )

    catalog = DatabaseNotesRepository(metadata_dir, "Unity", notes_root).load()

    assert catalog.notes[note_id].last_recorded_name == "Readable title"

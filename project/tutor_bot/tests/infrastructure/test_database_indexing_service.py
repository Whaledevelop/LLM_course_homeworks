import json
from pathlib import Path

from tutor_bot.infrastructure.database_indexing_service import DatabaseIndexingService
from tutor_bot.infrastructure.database_repository import DatabaseRepository


def test_indexes_nested_notes_and_migrates_frontmatter(tmp_path: Path) -> None:
    notes_root = tmp_path / "notes"
    metadata_dir = tmp_path / "metadata"
    nested_dir = notes_root / "nested"
    hidden_dir = notes_root / ".hidden"
    nested_dir.mkdir(parents=True)
    hidden_dir.mkdir()
    legacy_note = notes_root / "legacy.md"
    plain_note = nested_dir / "plain.md"
    legacy_note.write_text(
        "---\nid: 0b2c61a8-505a-552c-b731-7b9627970eff\ntitle: Legacy\ntag: x\n---\nBody\n"
    )
    plain_note.write_text("# Plain\n")
    (hidden_dir / "ignored.md").write_text("# Ignored\n")
    service = DatabaseIndexingService(DatabaseRepository(metadata_dir))

    result = service.update("Unity", notes_root)

    index = json.loads((notes_root / "tutor_bot_db_index_data.json").read_text())
    metadata = json.loads((metadata_dir / "Unity_metadata.json").read_text())
    assert result.added == 2
    assert len(index["notes"]) == 2
    assert set(index["notes"]) == set(metadata["notes"])
    assert all(note["fullness"] >= 1 for note in metadata["notes"].values())
    assert legacy_note.read_text() == (
        "---\ntutor_bot_note_id: 0b2c61a8-505a-552c-b731-7b9627970eff\n---\nBody\n"
    )
    assert "tutor_bot_note_id:" in plain_note.read_text()


def test_updates_moved_path_and_archives_deleted_metadata(tmp_path: Path) -> None:
    notes_root = tmp_path / "notes"
    metadata_dir = tmp_path / "metadata"
    notes_root.mkdir()
    note_path = notes_root / "note.md"
    note_path.write_text("# Note\n")
    repository = DatabaseRepository(metadata_dir)
    service = DatabaseIndexingService(repository)
    service.update("Unity", notes_root)
    first_index = repository.load_index(notes_root, "Unity")
    note_id = next(iter(first_index.notes))
    moved_path = notes_root / "nested" / "moved.md"
    moved_path.parent.mkdir()
    note_path.replace(moved_path)

    moved_result = service.update("Unity", notes_root)
    moved_path.unlink()
    deleted_result = service.update("Unity", notes_root)

    assert moved_result.moved == 1
    assert deleted_result.archived == 1
    assert note_id in repository.load_archive("Unity").notes
    assert note_id not in repository.load_metadata("Unity").notes

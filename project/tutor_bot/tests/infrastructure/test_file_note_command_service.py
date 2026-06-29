from pathlib import Path

import pytest

from note_command_test_support import (
    NOTE_ID,
    FailingMetadataRepository,
    create_create_command,
    create_delete_command,
    create_storage,
    create_update_command,
)
from tutor_bot.infrastructure.file_note_command_service import (
    FileNoteCommandService,
)
from tutor_bot.infrastructure.notes_metadata_repository import (
    NotesMetadataRepository,
)


def test_creates_markdown_and_metadata(
    tmp_path: Path,
) -> None:
    metadata_file, source_notes_dir, _ = create_storage(tmp_path)
    repository = NotesMetadataRepository(metadata_file)
    service = FileNoteCommandService(repository, source_notes_dir)

    created_note = service.create_note(create_create_command())

    saved_catalog = repository.load()
    saved_metadata = saved_catalog.notes[created_note.id]
    note_path = source_notes_dir / saved_metadata.relative_path
    saved_markdown = note_path.read_text(encoding="utf-8")

    assert created_note.title == "New note"
    assert saved_metadata.relative_path.as_posix() == (f"_tutor_bot/{created_note.id}.md")
    assert saved_metadata.last_recorded_name == "New note"
    assert note_path.is_file()
    assert f"id: {created_note.id}" in saved_markdown
    assert "# New content" in saved_markdown
    assert len(list((tmp_path / "backups").glob("*.json"))) == 1


def test_removes_created_markdown_when_metadata_save_fails(
    tmp_path: Path,
) -> None:
    metadata_file, source_notes_dir, _ = create_storage(tmp_path)
    repository = FailingMetadataRepository(metadata_file)
    service = FileNoteCommandService(repository, source_notes_dir)

    with pytest.raises(RuntimeError, match="Metadata save failed"):
        service.create_note(create_create_command())

    created_files = list((source_notes_dir / "_tutor_bot").glob("*.md"))
    saved_catalog = repository.load()

    assert created_files == []
    assert len(saved_catalog.notes) == 1


def test_updates_markdown_and_metadata(
    tmp_path: Path,
) -> None:
    metadata_file, source_notes_dir, note_path = create_storage(tmp_path)
    repository = NotesMetadataRepository(metadata_file)
    service = FileNoteCommandService(repository, source_notes_dir)

    updated_note = service.update_note(create_update_command())

    saved_catalog = repository.load()
    saved_metadata = saved_catalog.notes[NOTE_ID]
    saved_markdown = note_path.read_text(encoding="utf-8")

    assert updated_note.title == "Updated GC"
    assert saved_metadata.last_recorded_name == "Updated GC"
    assert saved_metadata.mastery == 4
    assert "# Updated content" in saved_markdown
    assert f"id: {NOTE_ID}" in saved_markdown
    assert len(list((tmp_path / "backups").glob("*.json"))) == 1


def test_restores_markdown_when_metadata_save_fails(
    tmp_path: Path,
) -> None:
    metadata_file, source_notes_dir, note_path = create_storage(tmp_path)
    original_content = note_path.read_text(encoding="utf-8")

    repository = FailingMetadataRepository(metadata_file)
    service = FileNoteCommandService(repository, source_notes_dir)

    with pytest.raises(RuntimeError, match="Metadata save failed"):
        service.update_note(create_update_command())

    assert note_path.read_text(encoding="utf-8") == original_content


def test_deletes_markdown_and_metadata(
    tmp_path: Path,
) -> None:
    metadata_file, source_notes_dir, note_path = create_storage(tmp_path)
    repository = NotesMetadataRepository(metadata_file)
    service = FileNoteCommandService(repository, source_notes_dir)

    deleted_note = service.delete_note(create_delete_command())

    saved_catalog = repository.load()
    deleted_files = list(note_path.parent.glob("*.deleted"))

    assert deleted_note.id == NOTE_ID
    assert deleted_note.title == "GC"
    assert not note_path.exists()
    assert NOTE_ID not in saved_catalog.notes
    assert deleted_files == []
    assert len(list((tmp_path / "backups").glob("*.json"))) == 1


def test_restores_markdown_when_delete_metadata_save_fails(
    tmp_path: Path,
) -> None:
    metadata_file, source_notes_dir, note_path = create_storage(tmp_path)
    original_content = note_path.read_text(encoding="utf-8")

    repository = FailingMetadataRepository(metadata_file)
    service = FileNoteCommandService(repository, source_notes_dir)

    with pytest.raises(RuntimeError, match="Metadata save failed"):
        service.delete_note(create_delete_command())

    saved_catalog = repository.load()
    deleted_files = list(note_path.parent.glob("*.deleted"))

    assert note_path.read_text(encoding="utf-8") == original_content
    assert NOTE_ID in saved_catalog.notes
    assert deleted_files == []

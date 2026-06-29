from pathlib import Path
from uuid import UUID

import pytest

from note_command_test_support import (
    FailingMetadataRepository,
    NOTE_ID,
    create_storage,
)
from tutor_bot.infrastructure.metadata_synchronization_planner import (
    MetadataSynchronizationPlanner,
)
from tutor_bot.infrastructure.metadata_synchronization_service import (
    MetadataSynchronizationService,
)
from tutor_bot.infrastructure.note_consistency_checker import (
    NoteConsistencyChecker,
)
from tutor_bot.infrastructure.notes_metadata_repository import (
    NotesMetadataRepository,
)


_ORPHAN_NOTE_ID = UUID("5dcd7a09-d513-48a8-b90d-b1f42662b18a")


def test_returns_existing_catalog_without_backup(
    tmp_path: Path,
) -> None:
    metadata_file, source_notes_dir, _ = create_storage(tmp_path)
    repository = NotesMetadataRepository(metadata_file)
    original_catalog = repository.load()
    service = _create_service(
        repository,
        source_notes_dir,
    )

    synchronized_catalog = service.synchronize()

    assert synchronized_catalog == original_catalog
    assert not (tmp_path / "backups").exists()


def test_removes_stale_metadata_and_registers_markdown(
    tmp_path: Path,
) -> None:
    metadata_file, source_notes_dir, note_path = create_storage(tmp_path)
    original_metadata = metadata_file.read_text(encoding="utf-8")
    note_path.unlink()

    orphan_note_path = source_notes_dir / "orphan.md"
    _write_markdown(
        orphan_note_path,
        _ORPHAN_NOTE_ID,
    )

    repository = NotesMetadataRepository(metadata_file)
    service = _create_service(
        repository,
        source_notes_dir,
    )

    synchronized_catalog = service.synchronize()
    synchronized_metadata = synchronized_catalog.notes[_ORPHAN_NOTE_ID]
    backup_files = list((tmp_path / "backups").glob("*.json"))

    assert NOTE_ID not in synchronized_catalog.notes
    assert synchronized_metadata.last_recorded_name == "orphan"
    assert synchronized_metadata.relative_path.as_posix() == "orphan.md"
    assert synchronized_metadata.theme == ""
    assert synchronized_metadata.importance == 0
    assert len(backup_files) == 1
    assert backup_files[0].read_text(encoding="utf-8") == original_metadata

    report = NoteConsistencyChecker(
        repository,
        source_notes_dir,
    ).check()

    assert report.is_consistent


def test_blocks_unsafe_synchronization(
    tmp_path: Path,
) -> None:
    metadata_file, source_notes_dir, note_path = create_storage(tmp_path)
    original_metadata = metadata_file.read_text(encoding="utf-8")

    _write_markdown(
        note_path,
        _ORPHAN_NOTE_ID,
    )

    repository = NotesMetadataRepository(metadata_file)
    service = _create_service(
        repository,
        source_notes_dir,
    )

    with pytest.raises(
        RuntimeError,
        match="Metadata synchronization is blocked",
    ):
        service.synchronize()

    assert metadata_file.read_text(encoding="utf-8") == original_metadata
    assert not (tmp_path / "backups").exists()


def test_preserves_files_when_metadata_save_fails(
    tmp_path: Path,
) -> None:
    metadata_file, source_notes_dir, note_path = create_storage(tmp_path)
    original_metadata = metadata_file.read_text(encoding="utf-8")
    note_path.unlink()

    orphan_note_path = source_notes_dir / "orphan.md"
    _write_markdown(
        orphan_note_path,
        _ORPHAN_NOTE_ID,
    )

    repository = FailingMetadataRepository(metadata_file)
    service = _create_service(
        repository,
        source_notes_dir,
    )

    with pytest.raises(
        RuntimeError,
        match="Metadata save failed",
    ):
        service.synchronize()

    assert metadata_file.read_text(encoding="utf-8") == original_metadata
    assert orphan_note_path.is_file()


def _create_service(
    repository: NotesMetadataRepository,
    source_notes_dir: Path,
) -> MetadataSynchronizationService:
    checker = NoteConsistencyChecker(
        repository,
        source_notes_dir,
    )
    planner = MetadataSynchronizationPlanner(checker)

    return MetadataSynchronizationService(
        repository,
        planner,
        source_notes_dir,
    )


def _write_markdown(
    note_path: Path,
    note_id: UUID,
) -> None:
    note_path.write_text(
        f"---\nid: {note_id}\n---\n\n# Test note\n",
        encoding="utf-8",
    )

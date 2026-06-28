import json
from pathlib import Path
from uuid import UUID

import pytest

from tutor_bot.application.update_note_command import UpdateNoteCommand
from tutor_bot.infrastructure.file_note_command_service import (
    FileNoteCommandService,
)
from tutor_bot.infrastructure.notes_metadata_repository import (
    NotesMetadataRepository,
)
from tutor_bot.schemas.notes_metadata_catalog import NotesMetadataCatalog


_NOTE_ID = UUID("2e2a0b1a-43f0-5d43-918f-393d557d5eac")


class _FailingMetadataRepository(NotesMetadataRepository):
    def save(
        self,
        catalog: NotesMetadataCatalog,
    ) -> NotesMetadataCatalog:
        raise RuntimeError("Metadata save failed")


def test_updates_markdown_and_metadata(
    tmp_path: Path,
) -> None:
    metadata_file, source_notes_dir, note_path = _create_storage(tmp_path)
    repository = NotesMetadataRepository(metadata_file)
    service = FileNoteCommandService(
        repository,
        source_notes_dir,
    )

    updated_note = service.update_note(_create_command())

    saved_catalog = repository.load()
    saved_metadata = saved_catalog.notes[_NOTE_ID]
    saved_markdown = note_path.read_text(encoding="utf-8")

    assert updated_note.title == "Updated GC"
    assert saved_metadata.last_recorded_name == "Updated GC"
    assert saved_metadata.mastery == 4
    assert "# Updated content" in saved_markdown
    assert f"id: {_NOTE_ID}" in saved_markdown
    assert len(list((tmp_path / "backups").glob("*.json"))) == 1


def test_restores_markdown_when_metadata_save_fails(
    tmp_path: Path,
) -> None:
    metadata_file, source_notes_dir, note_path = _create_storage(tmp_path)
    original_content = note_path.read_text(encoding="utf-8")

    repository = _FailingMetadataRepository(metadata_file)
    service = FileNoteCommandService(
        repository,
        source_notes_dir,
    )

    with pytest.raises(RuntimeError, match="Metadata save failed"):
        service.update_note(_create_command())

    assert note_path.read_text(encoding="utf-8") == original_content


def _create_storage(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    source_notes_dir = tmp_path / "notes"
    note_path = source_notes_dir / "csharp" / "gc.md"
    note_path.parent.mkdir(parents=True)

    note_path.write_text(
        f"---\nid: {_NOTE_ID}\n---\n\n# Original content\n",
        encoding="utf-8",
    )

    metadata_file = tmp_path / "notes_metadata.json"
    metadata_file.write_text(
        json.dumps(
            {
                "version": 2,
                "version_time": "2026-06-28T22:43:52+03:00",
                "notes": {
                    str(_NOTE_ID): {
                        "theme": "csharp",
                        "comment": "repeat",
                        "difficulty": "middle",
                        "importance": 8,
                        "completeness": 7,
                        "mastery": 1,
                        "last_recorded_name": "GC",
                        "relative_path": "csharp/gc.md",
                    }
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return metadata_file, source_notes_dir, note_path


def _create_command() -> UpdateNoteCommand:
    return UpdateNoteCommand(
        note_id=_NOTE_ID,
        title="Updated GC",
        theme="csharp",
        comment="ready",
        difficulty="middle",
        importance=9,
        completeness=8,
        mastery=4,
        markdown_content="# Updated content",
    )

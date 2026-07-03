import json
from pathlib import Path
from uuid import UUID

from tutor_bot.application.create_note_command import CreateNoteCommand
from tutor_bot.application.delete_note_command import DeleteNoteCommand
from tutor_bot.application.update_note_command import UpdateNoteCommand
from tutor_bot.infrastructure.notes_metadata_repository import (
    NotesMetadataRepository,
)
from tutor_bot.schemas.notes_metadata_catalog import NotesMetadataCatalog


NOTE_ID = UUID("2e2a0b1a-43f0-5d43-918f-393d557d5eac")


class FailingMetadataRepository(NotesMetadataRepository):
    def save(
        self,
        catalog: NotesMetadataCatalog,
    ) -> NotesMetadataCatalog:
        raise RuntimeError("Metadata save failed")


def create_storage(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    source_notes_dir = tmp_path / "notes"
    note_path = source_notes_dir / "csharp" / "gc.md"
    note_path.parent.mkdir(parents=True)

    note_path.write_text(
        f"---\ntutor_bot_note_id: {NOTE_ID}\n---\n\n# Original content\n",
        encoding="utf-8",
    )

    metadata_file = tmp_path / "notes_metadata.json"
    metadata_file.write_text(
        json.dumps(
            {
                "version": 2,
                "version_time": "2026-06-28T22:43:52+03:00",
                "notes": {
                    str(NOTE_ID): {
                        "group": "csharp",
                        "comment": "repeat",
                        "importance": 8,
                        "knowledge": 1,
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


def create_create_command() -> CreateNoteCommand:
    return CreateNoteCommand(
        title="New note",
        group="csharp",
        comment="learn",
        importance=7,
        knowledge=0,
        markdown_content="# New content",
    )


def create_update_command() -> UpdateNoteCommand:
    return UpdateNoteCommand(
        note_id=NOTE_ID,
        title="Updated GC",
        group="csharp",
        comment="ready",
        importance=9,
        knowledge=4,
        markdown_content="# Updated content",
    )


def create_delete_command() -> DeleteNoteCommand:
    return DeleteNoteCommand(note_id=NOTE_ID)

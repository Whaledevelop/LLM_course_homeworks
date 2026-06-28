import json
from pathlib import Path
from uuid import UUID

from tutor_bot.infrastructure.metadata_note_query_service import (
    MetadataNoteQueryService,
)
from tutor_bot.infrastructure.notes_metadata_repository import (
    NotesMetadataRepository,
)


def test_lists_metadata_notes_sorted_by_title(
    tmp_path: Path,
) -> None:
    metadata_file = tmp_path / "notes_metadata.json"

    metadata = {
        "version": 2,
        "version_time": "2026-06-28T22:43:52+03:00",
        "notes": {
            "e248dd6f-963d-552f-8616-d87082905b4e": {
                "theme": "threads",
                "comment": "learn",
                "difficulty": "middle",
                "importance": 9,
                "completeness": 9,
                "mastery": 2,
                "last_recorded_name": "Бета",
                "relative_path": "threads/beta.md",
            },
            "2e2a0b1a-43f0-5d43-918f-393d557d5eac": {
                "theme": "csharp",
                "comment": "repeat",
                "difficulty": "middle",
                "importance": 8,
                "completeness": 7,
                "mastery": 1,
                "last_recorded_name": "Альфа",
                "relative_path": "csharp/alpha.md",
            },
        },
    }

    metadata_file.write_text(
        json.dumps(metadata, ensure_ascii=False),
        encoding="utf-8-sig",
    )

    metadata_repository = NotesMetadataRepository(metadata_file)
    query_service = MetadataNoteQueryService(
        metadata_repository,
        tmp_path / "notes",
    )

    notes = query_service.list_notes()

    assert [note.title for note in notes] == ["Альфа", "Бета"]
    assert notes[0].theme == "csharp"
    assert notes[0].importance == 8
    assert str(notes[0].id) == "2e2a0b1a-43f0-5d43-918f-393d557d5eac"


def test_get_note_returns_markdown_without_frontmatter(
    tmp_path: Path,
) -> None:
    note_id = UUID("2e2a0b1a-43f0-5d43-918f-393d557d5eac")
    source_notes_dir = tmp_path / "notes"
    note_path = source_notes_dir / "csharp" / "garbage-collector.md"
    note_path.parent.mkdir(parents=True)

    note_path.write_text(
        (f"---\nid: {note_id}\n---\n\n# Garbage collector\n\nGC автоматически управляет памятью."),
        encoding="utf-8",
    )

    metadata = {
        "version": 2,
        "version_time": "2026-06-28T22:43:52+03:00",
        "notes": {
            str(note_id): {
                "theme": "csharp",
                "comment": "repeat",
                "difficulty": "middle",
                "importance": 8,
                "completeness": 7,
                "mastery": 1,
                "last_recorded_name": "Garbage collector",
                "relative_path": "csharp/garbage-collector.md",
            }
        },
    }

    metadata_file = tmp_path / "notes_metadata.json"
    metadata_file.write_text(
        json.dumps(metadata, ensure_ascii=False),
        encoding="utf-8-sig",
    )

    query_service = MetadataNoteQueryService(
        NotesMetadataRepository(metadata_file),
        source_notes_dir,
    )

    note = query_service.get_note(note_id)

    assert note.title == "Garbage collector"
    assert note.markdown_content.startswith("# Garbage collector")
    assert "id:" not in note.markdown_content

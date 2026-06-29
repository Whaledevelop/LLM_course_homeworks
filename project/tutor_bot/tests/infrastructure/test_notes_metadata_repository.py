import json
from pathlib import Path

from tutor_bot.infrastructure.notes_metadata_repository import (
    NotesMetadataRepository,
)


def test_saves_metadata_atomically_with_backup(
    tmp_path: Path,
) -> None:
    metadata_file = tmp_path / "notes_metadata.json"

    original_metadata = {
        "version": 2,
        "version_time": "2026-06-28T22:43:52+03:00",
        "notes": {
            "2e2a0b1a-43f0-5d43-918f-393d557d5eac": {
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

    original_content = json.dumps(
        original_metadata,
        ensure_ascii=False,
        indent=2,
    )

    metadata_file.write_text(
        original_content,
        encoding="utf-8",
    )

    repository = NotesMetadataRepository(metadata_file)
    catalog = repository.load()
    updated_catalog = repository.save(catalog)

    saved_catalog = repository.load()
    saved_content = metadata_file.read_text(encoding="utf-8")
    saved_metadata = next(iter(saved_catalog.notes.values()))

    backup_files = list((tmp_path / "backups").glob("*.json"))
    temporary_files = list(tmp_path.glob("*.tmp"))

    assert saved_catalog == updated_catalog
    assert saved_catalog.version_time > catalog.version_time
    assert len(saved_catalog.notes) == 1
    assert saved_metadata.relative_path.as_posix() == ("csharp/garbage-collector.md")
    assert '"relative_path": "csharp/garbage-collector.md"' in saved_content
    assert len(backup_files) == 1
    assert backup_files[0].read_text(encoding="utf-8") == original_content
    assert temporary_files == []

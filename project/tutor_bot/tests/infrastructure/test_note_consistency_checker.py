from pathlib import Path, PurePosixPath
from uuid import UUID

from note_command_test_support import NOTE_ID, create_storage
from tutor_bot.infrastructure.note_consistency_checker import (
    NoteConsistencyChecker,
)
from tutor_bot.infrastructure.notes_metadata_repository import (
    NotesMetadataRepository,
)
from tutor_bot.schemas.notes_metadata_catalog import NotesMetadataCatalog


_ORPHAN_NOTE_ID = UUID("5dcd7a09-d513-48a8-b90d-b1f42662b18a")
_SECOND_NOTE_ID = UUID("3ab8a412-b6bb-41cb-a241-767b50dd6174")


class _CatalogRepository:
    def __init__(
        self,
        catalog: NotesMetadataCatalog,
    ) -> None:
        self._catalog = catalog

    def load(self) -> NotesMetadataCatalog:
        return self._catalog


def test_reports_consistent_storage(
    tmp_path: Path,
) -> None:
    metadata_file, source_notes_dir, _ = create_storage(tmp_path)
    checker = NoteConsistencyChecker(
        NotesMetadataRepository(metadata_file),
        source_notes_dir,
    )

    report = checker.check()

    assert report.is_consistent
    assert report.missing_markdown_note_ids == ()
    assert report.markdown_without_metadata == ()
    assert report.duplicate_markdown_note_ids == ()
    assert report.duplicate_metadata_paths == ()
    assert report.frontmatter_id_mismatches == ()
    assert report.invalid_markdown_paths == ()


def test_reports_missing_and_invalid_markdown(
    tmp_path: Path,
) -> None:
    metadata_file, source_notes_dir, note_path = create_storage(tmp_path)
    note_path.unlink()

    invalid_note_path = source_notes_dir / "invalid.md"
    invalid_note_path.write_text(
        "# Missing frontmatter\n",
        encoding="utf-8",
    )

    checker = NoteConsistencyChecker(
        NotesMetadataRepository(metadata_file),
        source_notes_dir,
    )

    report = checker.check()

    assert not report.is_consistent
    assert report.missing_markdown_note_ids == (NOTE_ID,)
    assert report.invalid_markdown_paths == (PurePosixPath("invalid.md"),)


def test_reports_markdown_without_metadata_and_duplicate_ids(
    tmp_path: Path,
) -> None:
    metadata_file, source_notes_dir, _ = create_storage(tmp_path)

    first_orphan_path = source_notes_dir / "orphan-a.md"
    second_orphan_path = source_notes_dir / "nested" / "orphan-b.md"
    second_orphan_path.parent.mkdir()

    _write_markdown(
        first_orphan_path,
        _ORPHAN_NOTE_ID,
    )
    _write_markdown(
        second_orphan_path,
        _ORPHAN_NOTE_ID,
    )

    checker = NoteConsistencyChecker(
        NotesMetadataRepository(metadata_file),
        source_notes_dir,
    )

    report = checker.check()

    assert not report.is_consistent
    assert report.markdown_without_metadata == (
        PurePosixPath("nested/orphan-b.md"),
        PurePosixPath("orphan-a.md"),
    )
    assert report.duplicate_markdown_note_ids == (_ORPHAN_NOTE_ID,)


def test_reports_frontmatter_id_mismatch(
    tmp_path: Path,
) -> None:
    metadata_file, source_notes_dir, note_path = create_storage(tmp_path)

    _write_markdown(
        note_path,
        _ORPHAN_NOTE_ID,
    )

    checker = NoteConsistencyChecker(
        NotesMetadataRepository(metadata_file),
        source_notes_dir,
    )

    report = checker.check()

    assert not report.is_consistent
    assert report.markdown_without_metadata == (PurePosixPath("csharp/gc.md"),)
    assert report.frontmatter_id_mismatches == (PurePosixPath("csharp/gc.md"),)


def test_reports_duplicate_metadata_paths(
    tmp_path: Path,
) -> None:
    metadata_file, source_notes_dir, _ = create_storage(tmp_path)
    catalog = NotesMetadataRepository(metadata_file).load()
    metadata = catalog.notes[NOTE_ID]

    duplicate_catalog = catalog.model_construct(
        version=catalog.version,
        version_time=catalog.version_time,
        notes={
            NOTE_ID: metadata,
            _SECOND_NOTE_ID: metadata.model_copy(),
        },
    )

    checker = NoteConsistencyChecker(
        _CatalogRepository(duplicate_catalog),
        source_notes_dir,
    )

    report = checker.check()

    assert not report.is_consistent
    assert report.duplicate_metadata_paths == (PurePosixPath("csharp/gc.md"),)


def _write_markdown(
    note_path: Path,
    note_id: UUID,
) -> None:
    note_path.write_text(
        f"---\ntutor_bot_note_id: {note_id}\n---\n\n# Test note\n",
        encoding="utf-8",
    )

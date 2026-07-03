from pathlib import Path, PurePosixPath
from uuid import UUID

from note_command_test_support import NOTE_ID, create_storage
from tutor_bot.infrastructure.metadata_synchronization_planner import (
    MetadataSynchronizationPlanner,
)
from tutor_bot.infrastructure.note_consistency_checker import (
    NoteConsistencyChecker,
)
from tutor_bot.infrastructure.notes_metadata_repository import (
    NotesMetadataRepository,
)


_ORPHAN_NOTE_ID = UUID("5dcd7a09-d513-48a8-b90d-b1f42662b18a")


def test_creates_empty_plan_for_consistent_storage(
    tmp_path: Path,
) -> None:
    metadata_file, source_notes_dir, _ = create_storage(tmp_path)
    planner = _create_planner(
        metadata_file,
        source_notes_dir,
    )

    plan = planner.create_plan()

    assert plan.can_apply
    assert not plan.has_changes
    assert plan.metadata_note_ids_to_remove == ()
    assert plan.markdown_paths_to_register == ()


def test_plans_repairable_metadata_changes(
    tmp_path: Path,
) -> None:
    metadata_file, source_notes_dir, note_path = create_storage(tmp_path)
    note_path.unlink()

    orphan_note_path = source_notes_dir / "orphan.md"
    _write_markdown(
        orphan_note_path,
        _ORPHAN_NOTE_ID,
    )

    planner = _create_planner(
        metadata_file,
        source_notes_dir,
    )

    plan = planner.create_plan()

    assert plan.can_apply
    assert plan.has_changes
    assert plan.metadata_note_ids_to_remove == (NOTE_ID,)
    assert plan.markdown_paths_to_register == (PurePosixPath("orphan.md"),)


def test_blocks_plan_with_unsafe_issues(
    tmp_path: Path,
) -> None:
    metadata_file, source_notes_dir, note_path = create_storage(tmp_path)

    _write_markdown(
        note_path,
        _ORPHAN_NOTE_ID,
    )

    invalid_note_path = source_notes_dir / "invalid.md"
    invalid_note_path.write_text(
        "# Missing frontmatter\n",
        encoding="utf-8",
    )

    planner = _create_planner(
        metadata_file,
        source_notes_dir,
    )

    plan = planner.create_plan()

    assert not plan.can_apply
    assert plan.consistency_report.frontmatter_id_mismatches == (PurePosixPath("csharp/gc.md"),)
    assert plan.consistency_report.invalid_markdown_paths == (PurePosixPath("invalid.md"),)


def _create_planner(
    metadata_file: Path,
    source_notes_dir: Path,
) -> MetadataSynchronizationPlanner:
    repository = NotesMetadataRepository(metadata_file)
    checker = NoteConsistencyChecker(
        repository,
        source_notes_dir,
    )

    return MetadataSynchronizationPlanner(checker)


def _write_markdown(
    note_path: Path,
    note_id: UUID,
) -> None:
    note_path.write_text(
        f"---\ntutor_bot_note_id: {note_id}\n---\n\n# Test note\n",
        encoding="utf-8",
    )

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from note_command_test_support import NOTE_ID, create_storage
from tutor_bot.infrastructure.notes_metadata_repository import (
    NotesMetadataRepository,
)
from tutor_bot.synchronize_metadata import main


_ORPHAN_NOTE_ID = UUID("5dcd7a09-d513-48a8-b90d-b1f42662b18a")


def test_returns_zero_for_consistent_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata_file, source_notes_dir, _ = create_storage(tmp_path)
    _configure_settings(
        monkeypatch,
        metadata_file,
        source_notes_dir,
    )

    exit_code = main([])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["mode"] == "dry-run"
    assert not output["has_changes"]
    assert output["can_apply"]
    assert not (tmp_path / "backups").exists()


def test_dry_run_and_apply_repairable_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata_file, source_notes_dir, note_path = create_storage(tmp_path)
    original_metadata = metadata_file.read_text(encoding="utf-8")
    note_path.unlink()

    orphan_note_path = source_notes_dir / "orphan.md"
    _write_markdown(
        orphan_note_path,
        _ORPHAN_NOTE_ID,
    )

    _configure_settings(
        monkeypatch,
        metadata_file,
        source_notes_dir,
    )

    dry_run_exit_code = main([])
    dry_run_output = json.loads(capsys.readouterr().out)

    assert dry_run_exit_code == 1
    assert dry_run_output["mode"] == "dry-run"
    assert dry_run_output["has_changes"]
    assert dry_run_output["can_apply"]
    assert metadata_file.read_text(encoding="utf-8") == original_metadata
    assert not (tmp_path / "backups").exists()

    apply_exit_code = main(["--apply"])
    apply_output = capsys.readouterr().out
    saved_catalog = NotesMetadataRepository(metadata_file).load()
    backup_files = list((tmp_path / "backups").glob("*.json"))

    assert apply_exit_code == 0
    assert '"mode": "apply"' in apply_output
    assert '"applied": true' in apply_output
    assert '"is_consistent": true' in apply_output
    assert NOTE_ID not in saved_catalog.notes
    assert _ORPHAN_NOTE_ID in saved_catalog.notes
    assert len(backup_files) == 1


def test_returns_two_for_blocking_issues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata_file, source_notes_dir, note_path = create_storage(tmp_path)
    original_metadata = metadata_file.read_text(encoding="utf-8")

    _write_markdown(
        note_path,
        _ORPHAN_NOTE_ID,
    )

    _configure_settings(
        monkeypatch,
        metadata_file,
        source_notes_dir,
    )

    exit_code = main([])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert output["mode"] == "dry-run"
    assert not output["can_apply"]
    assert metadata_file.read_text(encoding="utf-8") == original_metadata
    assert not (tmp_path / "backups").exists()


def _configure_settings(
    monkeypatch: pytest.MonkeyPatch,
    metadata_file: Path,
    source_notes_dir: Path,
) -> None:
    settings = SimpleNamespace(
        metadata_file=metadata_file,
        source_notes_dir=source_notes_dir,
    )

    monkeypatch.setattr(
        "tutor_bot.synchronize_metadata.get_settings",
        lambda: settings,
    )


def _write_markdown(
    note_path: Path,
    note_id: UUID,
) -> None:
    note_path.write_text(
        f"---\ntutor_bot_note_id: {note_id}\n---\n\n# Test note\n",
        encoding="utf-8",
    )

from pathlib import Path
from uuid import UUID

from tutor_bot.infrastructure.ui_state_repository import UiStateRepository


def test_returns_none_when_ui_state_file_is_missing(
    tmp_path: Path,
) -> None:
    repository = UiStateRepository(tmp_path / "ui_state.json")

    assert repository.load_selected_note_id() is None


def test_saves_and_loads_selected_note_id(
    tmp_path: Path,
) -> None:
    note_id = UUID("2e2a0b1a-43f0-5d43-918f-393d557d5eac")
    repository = UiStateRepository(tmp_path / "ui_state.json")

    repository.save_selected_note_id(note_id)

    assert repository.load_selected_note_id() == note_id

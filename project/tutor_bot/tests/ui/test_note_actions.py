from unittest.mock import Mock
from uuid import uuid4

from tutor_bot.application.note_details import NoteDetails
from tutor_bot.ui.views import note_actions


def test_repeated_edit_click_requests_save(monkeypatch) -> None:
    note_details = NoteDetails(
        id=uuid4(),
        title="Заметка",
        group="Группа",
        comment="",
        importance=5,
        knowledge=5,
        fullness=5,
        markdown_content="Текст",
    )
    note_id = str(note_details.id)
    edit_column = Mock()
    edit_column.button.return_value = True
    test_column = Mock()
    delete_column = Mock()
    delete_column.button.return_value = False
    session_state = {note_actions._ACTIVE_EDITOR_KEY: note_id}
    render_edit_form = Mock(return_value="Заметка сохранена.")

    monkeypatch.setattr(
        note_actions.st, "columns", Mock(return_value=[edit_column, test_column, delete_column])
    )
    monkeypatch.setattr(note_actions.st, "session_state", session_state)
    monkeypatch.setattr(note_actions, "_render_edit_form", render_edit_form)

    result = note_actions.render_note_actions(
        note_details,
        Mock(),
        Mock(),
        Mock(),
    )

    assert result == "Заметка сохранена."
    assert session_state[note_actions._ACTIVE_EDITOR_KEY] == note_id
    assert render_edit_form.call_args.args[-1] is True


def test_first_edit_click_only_opens_editor(monkeypatch) -> None:
    note_details = NoteDetails(
        id=uuid4(),
        title="Заметка",
        group="Группа",
        comment="",
        importance=5,
        knowledge=5,
        fullness=5,
        markdown_content="Текст",
    )
    edit_column = Mock()
    edit_column.button.return_value = True
    test_column = Mock()
    delete_column = Mock()
    delete_column.button.return_value = False
    session_state = {}
    render_edit_form = Mock(return_value=None)

    monkeypatch.setattr(
        note_actions.st,
        "columns",
        Mock(return_value=[edit_column, test_column, delete_column]),
    )
    monkeypatch.setattr(note_actions.st, "session_state", session_state)
    monkeypatch.setattr(note_actions, "_render_edit_form", render_edit_form)

    note_actions.render_note_actions(
        note_details,
        Mock(),
        Mock(),
        Mock(),
    )

    assert session_state[note_actions._ACTIVE_EDITOR_KEY] == str(note_details.id)
    assert render_edit_form.call_args.args[-1] is False

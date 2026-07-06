from tutor_bot.ui.app_mode import APP_MODE_STATE_KEY, AppMode
from tutor_bot.ui.views import databases_page


def test_clear_note_session_state_preserves_selected_app_mode(monkeypatch) -> None:
    session_state = {
        APP_MODE_STATE_KEY: AppMode.DATABASES,
        "selected_note_id": "note-1",
    }
    monkeypatch.setattr(databases_page.st, "session_state", session_state)

    databases_page._clear_note_session_state()

    assert session_state == {APP_MODE_STATE_KEY: AppMode.DATABASES}

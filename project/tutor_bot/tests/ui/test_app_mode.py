from tutor_bot.ui import app_mode
from tutor_bot.ui.app_mode import APP_MODE_STATE_KEY, AppMode


def test_requested_app_mode_is_applied_before_widget_creation(monkeypatch) -> None:
    session_state = {APP_MODE_STATE_KEY: AppMode.QUESTIONS}
    monkeypatch.setattr(app_mode.st, "session_state", session_state)

    app_mode.request_app_mode(AppMode.TEST_NOTES)

    assert session_state[APP_MODE_STATE_KEY] == AppMode.QUESTIONS

    app_mode.apply_pending_app_mode()

    assert session_state == {APP_MODE_STATE_KEY: AppMode.TEST_NOTES}

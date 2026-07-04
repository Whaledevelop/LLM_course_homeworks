from contextlib import nullcontext
from types import SimpleNamespace
from uuid import uuid4

from tutor_bot.ui.app_mode import APP_MODE_STATE_KEY, AppMode
from tutor_bot.ui.views import active_recall_page


def test_start_note_test_opens_created_study_session(monkeypatch) -> None:
    note_id = uuid4()
    study_session = object()
    recall_service = SimpleNamespace(
        create_note_study_session=lambda requested_note_id: study_session,
    )
    streamlit = SimpleNamespace(
        session_state={},
        spinner=lambda message: nullcontext(),
    )
    monkeypatch.setattr(active_recall_page, "st", streamlit)

    active_recall_page.start_note_test(recall_service, note_id)

    assert streamlit.session_state[active_recall_page._STUDY_SESSION_KEY] is study_session
    assert streamlit.session_state[APP_MODE_STATE_KEY] == AppMode.TEST_NOTES

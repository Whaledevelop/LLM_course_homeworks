import streamlit as st
from httpx import HTTPError
from pydantic import ValidationError
from uuid import UUID

from tutor_bot.application.active_recall_service import ActiveRecallService
from tutor_bot.application.note_query_service import NoteQueryService
from tutor_bot.application.recall_study_session import RecallStudySession
from tutor_bot.ui.app_mode import APP_MODE_STATE_KEY, AppMode
from tutor_bot.ui.views.active_recall_session_view import (
    render_active_recall_session,
)
from tutor_bot.ui.views.active_recall_history_page import render_active_recall_history_page


_STUDY_SESSION_KEY = "active_recall_study_session"
_SHOW_NOTE_BEFORE_KEY = "active_recall_show_note_before"
_SHOW_NOTE_AFTER_KEY = "active_recall_show_note_after"


def interrupt_active_recall_session() -> None:
    st.session_state.pop(_STUDY_SESSION_KEY, None)


def start_note_test(
    recall_service: ActiveRecallService,
    note_id: UUID,
) -> None:
    try:
        with st.spinner("Подготавливаю вопрос по заметке..."):
            st.session_state[_STUDY_SESSION_KEY] = recall_service.create_note_study_session(note_id)
    except (HTTPError, RuntimeError, ValueError, ValidationError) as error:
        st.error(f"Не удалось начать тест заметки: {error}")

        return

    st.session_state[APP_MODE_STATE_KEY] = AppMode.TEST_NOTES


def open_note_study_session(
    study_session: RecallStudySession,
) -> None:
    st.session_state[_STUDY_SESSION_KEY] = study_session
    st.session_state[APP_MODE_STATE_KEY] = AppMode.TEST_NOTES


def render_active_recall_page(
    note_query_service: NoteQueryService,
    recall_service: ActiveRecallService,
) -> None:
    notes = note_query_service.list_notes()
    eligible_notes = [note for note in notes if note.fullness >= 4]

    if not eligible_notes:
        st.info("Для Test Notes нужны заметки с заполненностью 4 или выше.")
        render_active_recall_history_page(recall_service)

        return

    study_session = st.session_state.get(_STUDY_SESSION_KEY)
    is_session_running = study_session is not None

    if not is_session_running:
        max_question_count = min(10, len(eligible_notes))
        question_count = st.slider(
            "Количество вопросов",
            min_value=1,
            max_value=max_question_count,
            value=max_question_count,
        )
        st.toggle(
            "Показать заметку перед вопросом",
            key=_SHOW_NOTE_BEFORE_KEY,
        )
        st.toggle(
            "Показать заметку после вопроса",
            key=_SHOW_NOTE_AFTER_KEY,
        )
        start_label = "Начать новую сессию" if study_session else "Начать сессию"
        st.caption("Test Note using Active Recall")

        if st.button(start_label, type="primary"):
            study_session = _create_study_session(
                recall_service,
                question_count,
            )

            if study_session is not None:
                st.rerun()

    if not is_session_running:
        st.divider()
        render_active_recall_history_page(recall_service)

    if study_session is None:
        return

    st.divider()
    render_active_recall_session(
        recall_service,
        study_session,
        _STUDY_SESSION_KEY,
        st.session_state.get(_SHOW_NOTE_BEFORE_KEY, False),
        st.session_state.get(_SHOW_NOTE_AFTER_KEY, False),
    )


def _create_study_session(
    recall_service: ActiveRecallService,
    question_count: int,
) -> RecallStudySession | None:
    try:
        with st.spinner("Генерирую первый вопрос..."):
            study_session = recall_service.create_study_session(question_count)
            st.session_state[_STUDY_SESSION_KEY] = study_session

            return study_session
    except (HTTPError, RuntimeError, ValueError, ValidationError) as error:
        st.error(f"Не удалось начать сессию: {error}")

        return None

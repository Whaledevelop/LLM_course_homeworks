import streamlit as st
from httpx import HTTPError
from pydantic import ValidationError

from tutor_bot.application.active_recall_service import ActiveRecallService
from tutor_bot.application.note_query_service import NoteQueryService
from tutor_bot.application.recall_study_session import RecallStudySession
from tutor_bot.ui.views.active_recall_session_view import (
    render_active_recall_session,
)


_STUDY_SESSION_KEY = "active_recall_study_session"
_SHOW_NOTE_BEFORE_KEY = "active_recall_show_note_before"
_SHOW_NOTE_AFTER_KEY = "active_recall_show_note_after"


def interrupt_active_recall_session() -> None:
    st.session_state.pop(_STUDY_SESSION_KEY, None)


def render_active_recall_page(
    note_query_service: NoteQueryService,
    recall_service: ActiveRecallService,
) -> None:
    notes = note_query_service.list_notes()

    if not notes:
        st.info("Заметки пока отсутствуют.")

        return

    study_session = st.session_state.get(_STUDY_SESSION_KEY)
    is_session_running = study_session is not None and not study_session.is_complete

    if not is_session_running:
        max_question_count = min(10, len(notes))
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

        if st.button(start_label, type="primary"):
            study_session = _create_study_session(
                recall_service,
                question_count,
            )

            if study_session is not None:
                st.rerun()

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

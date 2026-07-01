from uuid import UUID

import streamlit as st
from httpx import HTTPError
from ollama import ResponseError
from pydantic import ValidationError

from tutor_bot.application.active_recall_service import ActiveRecallService
from tutor_bot.application.note_query_service import NoteQueryService
from tutor_bot.application.recall_session_result import RecallSessionResult
from tutor_bot.application.recall_study_session import RecallStudySession
from tutor_bot.ui.views.active_recall_session_view import (
    VERDICT_LABELS,
    render_active_recall_session,
)


_STUDY_SESSION_KEY = "active_recall_study_session"


def render_active_recall_page(
    note_query_service: NoteQueryService,
    recall_service: ActiveRecallService,
) -> None:
    st.header("Active Recall")
    st.caption("Вспомните материал без подсказок, затем сравните ответ с критериями заметки.")
    _render_history(recall_service.get_history())

    notes = note_query_service.list_notes()

    if not notes:
        st.info("Заметки пока отсутствуют.")

        return

    study_session = st.session_state.get(_STUDY_SESSION_KEY)
    notes_by_id = {note.id: note for note in notes}
    selected_note_id = st.selectbox(
        "Заметка",
        options=list(notes_by_id),
        format_func=lambda note_id: notes_by_id[note_id].title,
        disabled=study_session is not None and not study_session.is_complete,
    )

    if study_session is None or study_session.is_complete:
        start_label = "Начать новую сессию" if study_session else "Начать сессию"

        if st.button(start_label, type="primary"):
            study_session = _create_study_session(
                recall_service,
                selected_note_id,
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
    )


def _create_study_session(
    recall_service: ActiveRecallService,
    note_id: UUID,
) -> RecallStudySession | None:
    try:
        with st.spinner("Генерирую первый вопрос..."):
            study_session = recall_service.create_study_session(note_id)
            st.session_state[_STUDY_SESSION_KEY] = study_session

            return study_session
    except (HTTPError, ResponseError, RuntimeError, ValueError, ValidationError) as error:
        st.error(f"Не удалось начать сессию: {error}")

        return None


def _render_history(
    history: tuple[RecallSessionResult, ...],
) -> None:
    if not history:
        return

    verdict_counts = {
        verdict: sum(result.review.verdict == verdict for result in history)
        for verdict in VERDICT_LABELS
    }
    columns = st.columns(4)
    columns[0].metric("Попыток", len(history))
    columns[1].metric("Правильных", verdict_counts["correct"])
    columns[2].metric("Частичных", verdict_counts["partially_correct"])
    columns[3].metric("Ошибочных", verdict_counts["incorrect"])

    with st.expander("Последние попытки"):
        for result in reversed(history[-5:]):
            st.markdown(
                f"**{result.session.note_title}** — {VERDICT_LABELS[result.review.verdict]}"
            )
            st.caption(result.session.exercise.question)

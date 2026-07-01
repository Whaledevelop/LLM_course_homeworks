from uuid import UUID

import streamlit as st
from httpx import HTTPError
from ollama import ResponseError
from pydantic import ValidationError

from tutor_bot.application.active_recall_service import ActiveRecallService
from tutor_bot.application.note_query_service import NoteQueryService
from tutor_bot.application.recall_session import RecallSession
from tutor_bot.application.recall_session_result import RecallSessionResult


_SESSION_KEY = "active_recall_session"
_RESULT_KEY = "active_recall_result"
_VERDICT_LABELS = {
    "correct": "Ответ корректный",
    "partially_correct": "Ответ частично корректный",
    "incorrect": "Ответ содержит существенные ошибки",
}


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

    notes_by_id = {note.id: note for note in notes}
    selected_note_id = st.selectbox(
        "Заметка",
        options=list(notes_by_id),
        format_func=lambda note_id: notes_by_id[note_id].title,
    )

    session = st.session_state.get(_SESSION_KEY)

    if session is not None and session.note_id != selected_note_id:
        st.session_state.pop(_SESSION_KEY, None)
        st.session_state.pop(_RESULT_KEY, None)
        session = None

    generate_label = "Сгенерировать новый вопрос" if session else "Сгенерировать вопрос"

    if st.button(generate_label, type="primary"):
        session = _create_session(
            recall_service,
            selected_note_id,
        )

    if session is None:
        return

    st.divider()
    st.caption(f"Заметка: {session.note_title}")
    st.subheader(session.exercise.question)

    result = st.session_state.get(_RESULT_KEY)

    if result is None:
        _render_answer_form(
            recall_service,
            session,
        )

        return

    _render_result(result)


def _create_session(
    recall_service: ActiveRecallService,
    note_id: UUID,
) -> RecallSession | None:
    st.session_state.pop(_RESULT_KEY, None)

    try:
        with st.spinner("Генерирую вопрос по заметке..."):
            session = recall_service.create_session(note_id)
            st.session_state[_SESSION_KEY] = session

            return session
    except (HTTPError, ResponseError, RuntimeError, ValidationError) as error:
        st.error(f"Не удалось создать упражнение: {error}")

        return None


def _render_answer_form(
    recall_service: ActiveRecallService,
    session: RecallSession,
) -> None:
    with st.form("active-recall-answer-form"):
        student_answer = st.text_area(
            "Ваш ответ",
            height=220,
        )
        submitted = st.form_submit_button(
            "Проверить ответ",
            type="primary",
        )

    if not submitted:
        return

    if not student_answer.strip():
        st.error("Введите ответ.")

        return

    try:
        with st.spinner("Проверяю раскрытые тезисы..."):
            st.session_state[_RESULT_KEY] = recall_service.review_answer(
                session,
                student_answer,
            )
        st.rerun()
    except (HTTPError, ResponseError, RuntimeError, ValidationError) as error:
        st.error(f"Не удалось проверить ответ: {error}")


def _render_result(result: RecallSessionResult) -> None:
    review = result.review
    verdict_label = _VERDICT_LABELS[review.verdict]

    if review.verdict == "correct":
        st.success(verdict_label)
    elif review.verdict == "partially_correct":
        st.warning(verdict_label)
    else:
        st.error(verdict_label)

    st.markdown(review.feedback)
    _render_points("Раскрытые тезисы", review.covered_points)
    _render_points("Пропущенные тезисы", review.missing_points)
    _render_points("Ошибки", review.errors)

    st.markdown("#### Эталонный ответ")
    st.markdown(result.session.exercise.reference_answer)


def _render_points(
    title: str,
    points: tuple[str, ...],
) -> None:
    if not points:
        return

    st.markdown(f"#### {title}")

    for point in points:
        st.markdown(f"- {point}")


def _render_history(
    history: tuple[RecallSessionResult, ...],
) -> None:
    if not history:
        return

    verdict_counts = {
        verdict: sum(result.review.verdict == verdict for result in history)
        for verdict in _VERDICT_LABELS
    }
    columns = st.columns(4)
    columns[0].metric("Попыток", len(history))
    columns[1].metric("Правильных", verdict_counts["correct"])
    columns[2].metric("Частичных", verdict_counts["partially_correct"])
    columns[3].metric("Ошибочных", verdict_counts["incorrect"])

    with st.expander("Последние попытки"):
        for result in reversed(history[-5:]):
            st.markdown(
                f"**{result.session.note_title}** — {_VERDICT_LABELS[result.review.verdict]}"
            )
            st.caption(result.session.exercise.question)

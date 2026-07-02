import streamlit as st
from httpx import HTTPError
from ollama import ResponseError
from pydantic import ValidationError

from tutor_bot.application.active_recall_service import ActiveRecallService
from tutor_bot.application.recall_session_result import RecallSessionResult
from tutor_bot.application.recall_study_session import RecallStudySession


VERDICT_LABELS = {
    "correct": "Ответ корректный",
    "partially_correct": "Ответ частично корректный",
    "incorrect": "Ответ содержит существенные ошибки",
}


def render_active_recall_session(
    recall_service: ActiveRecallService,
    study_session: RecallStudySession,
    state_key: str,
    show_note_before_question: bool,
    show_note_after_question: bool,
) -> None:
    if st.button("Прервать сессию"):
        st.session_state.pop(state_key, None)
        st.rerun()

    st.progress(
        study_session.answered_count / study_session.total_count,
        text=f"Отвечено: {study_session.answered_count}/{study_session.total_count}",
    )
    st.caption(
        f"Вопрос {study_session.current_index + 1}/{study_session.total_count} · "
        f"Заметка: {study_session.current_exercise.note_title}"
    )
    current_result = _get_current_result(study_session)

    if (
        current_result is None
        and show_note_before_question
        and study_session.current_exercise.source_markdown
    ):
        with st.expander(
            "Исходная заметка",
            expanded=True,
        ):
            st.markdown(study_session.current_exercise.source_markdown)

    st.subheader(study_session.current_exercise.exercise.question)

    if current_result is None:
        _render_answer_form(
            recall_service,
            study_session,
            state_key,
        )

        return

    _render_result(
        current_result,
        show_note_after_question,
    )

    if study_session.is_complete:
        _render_summary(study_session)

        return

    if st.button("Следующий вопрос", type="primary"):
        try:
            with st.spinner("Генерирую следующий вопрос..."):
                st.session_state[state_key] = recall_service.advance_study_session(study_session)
            st.rerun()
        except (HTTPError, ResponseError, RuntimeError, ValueError, ValidationError) as error:
            st.error(f"Не удалось перейти к следующему вопросу: {error}")


def _get_current_result(
    study_session: RecallStudySession,
) -> RecallSessionResult | None:
    if study_session.answered_count <= study_session.current_index:
        return None

    return study_session.results[study_session.current_index]


def _render_answer_form(
    recall_service: ActiveRecallService,
    study_session: RecallStudySession,
    state_key: str,
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
            st.session_state[state_key] = recall_service.review_study_answer(
                study_session,
                student_answer,
            )
        st.rerun()
    except (HTTPError, ResponseError, RuntimeError, ValueError, ValidationError) as error:
        st.error(f"Не удалось проверить ответ: {error}")


def _render_result(
    result: RecallSessionResult,
    show_note_after_question: bool,
) -> None:
    review = result.review
    verdict_label = VERDICT_LABELS[review.verdict]

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

    if show_note_after_question and result.session.source_markdown:
        with st.expander("Исходная заметка"):
            st.markdown(result.session.source_markdown)


def _render_points(
    title: str,
    points: tuple[str, ...],
) -> None:
    if not points:
        return

    st.markdown(f"#### {title}")

    for point in points:
        st.markdown(f"- {point}")


def _render_summary(study_session: RecallStudySession) -> None:
    verdict_counts = {
        verdict: sum(result.review.verdict == verdict for result in study_session.results)
        for verdict in VERDICT_LABELS
    }
    st.divider()
    st.subheader("Сессия завершена")
    columns = st.columns(3)
    columns[0].metric("Правильных", verdict_counts["correct"])
    columns[1].metric("Частичных", verdict_counts["partially_correct"])
    columns[2].metric("Ошибочных", verdict_counts["incorrect"])

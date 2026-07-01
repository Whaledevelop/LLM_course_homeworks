from uuid import UUID

import streamlit as st
from httpx import HTTPError
from ollama import ResponseError
from pydantic import ValidationError

from tutor_bot.application.assignment_review_result import AssignmentReviewResult
from tutor_bot.application.assignment_review_service import AssignmentReviewService
from tutor_bot.retrieval.hybrid_search_result import HybridSearchResult


_REVIEW_KEY = "assignment_review"
_VERDICT_LABELS = {
    "correct": "Ответ корректный",
    "partially_correct": "Ответ частично корректный",
    "incorrect": "Ответ содержит существенные ошибки",
    "insufficient_context": "Недостаточно материалов для проверки",
}


def render_assignment_review_page(
    review_service: AssignmentReviewService,
) -> None:
    st.header("Проверка заданий")
    st.caption("Ответ проверяется только по содержанию локальных заметок.")

    with st.form("assignment-review-form"):
        assignment_text = st.text_area(
            "Условие задания",
            placeholder="Опишите, что должен объяснить или решить ученик.",
            height=140,
        )
        student_answer = st.text_area(
            "Ответ ученика",
            placeholder="Вставьте ответ для проверки.",
            height=240,
        )
        submitted = st.form_submit_button(
            "Проверить ответ",
            type="primary",
        )

    if submitted:
        _submit_review(
            review_service,
            assignment_text,
            student_answer,
        )

    result = st.session_state.get(_REVIEW_KEY)

    if result is None:
        return

    _render_review(result)


def _submit_review(
    review_service: AssignmentReviewService,
    assignment_text: str,
    student_answer: str,
) -> None:
    if not assignment_text.strip() or not student_answer.strip():
        st.error("Заполните условие задания и ответ ученика.")

        return

    st.session_state.pop(
        _REVIEW_KEY,
        None,
    )

    try:
        with st.spinner("Ищу материалы и проверяю ответ..."):
            st.session_state[_REVIEW_KEY] = review_service.review(
                assignment_text,
                student_answer,
            )
    except (HTTPError, ResponseError, RuntimeError, ValidationError) as error:
        st.error(f"Не удалось проверить ответ: {error}")


def _render_review(result: AssignmentReviewResult) -> None:
    review = result.review
    verdict_label = _VERDICT_LABELS[review.verdict]

    st.divider()
    st.subheader("Результат проверки")

    if review.verdict == "correct":
        st.success(verdict_label)
    elif review.verdict == "partially_correct":
        st.warning(verdict_label)
    elif review.verdict == "incorrect":
        st.error(verdict_label)
    else:
        st.info(verdict_label)

    st.markdown(review.feedback)

    _render_points("Что раскрыто правильно", review.correct_points)
    _render_points("Ошибки", review.errors)
    _render_points("Что пропущено", review.missing_points)

    if not result.context.has_sufficient_context:
        return

    st.subheader("Использованные материалы")
    results_by_note: dict[UUID, list[HybridSearchResult]] = {}

    for context_result in result.context.selected_results:
        results_by_note.setdefault(
            context_result.chunk.note_id,
            [],
        ).append(context_result)

    for note_results in results_by_note.values():
        note = note_results[0].chunk

        with st.expander(f"{note.note_title} · {note.relative_path.as_posix()}"):
            for context_result in note_results:
                if context_result.chunk.heading_title:
                    st.caption(context_result.chunk.heading_title)

                st.markdown(context_result.chunk.text)


def _render_points(
    title: str,
    points: tuple[str, ...],
) -> None:
    if not points:
        return

    st.markdown(f"#### {title}")

    for point in points:
        st.markdown(f"- {point}")

from uuid import UUID

import streamlit as st
from httpx import HTTPError
from ollama import ResponseError
from openai import OpenAIError

from tutor_bot.application.tutor_answer import TutorAnswer
from tutor_bot.application.tutor_answer_service import TutorAnswerService
from tutor_bot.retrieval.hybrid_search_result import HybridSearchResult


_ANSWER_KEY = "tutor_answer"


def render_questions_page(
    answer_service: TutorAnswerService,
) -> None:
    st.header("Вопросы по материалам")
    st.caption("Ответ формируется только по найденным фрагментам локальных заметок.")

    with st.form("question-form"):
        question = st.text_area(
            "Вопрос",
            placeholder="Например: чем асинхронность отличается от многопоточности?",
            height=120,
        )
        submitted = st.form_submit_button(
            "Получить ответ",
            type="primary",
        )

    if submitted:
        _submit_question(
            answer_service,
            question,
        )

    answer = st.session_state.get(_ANSWER_KEY)

    if answer is None:
        return

    _render_answer(answer)


def _submit_question(
    answer_service: TutorAnswerService,
    question: str,
) -> None:
    if not question.strip():
        st.error("Введите вопрос.")

        return

    st.session_state.pop(
        _ANSWER_KEY,
        None,
    )

    try:
        with st.spinner("Ищу информацию в заметках и формирую ответ..."):
            st.session_state[_ANSWER_KEY] = answer_service.answer(question)
    except (HTTPError, OpenAIError, ResponseError) as error:
        st.error(f"Не удалось получить ответ от Ollama: {error}")


def _render_answer(answer: TutorAnswer) -> None:
    st.divider()
    st.subheader("Ответ")
    st.markdown(answer.answer)

    if not answer.context.has_sufficient_context:
        return

    st.subheader("Использованный контекст")

    results_by_note: dict[UUID, list[HybridSearchResult]] = {}

    for result in answer.context.selected_results:
        results_by_note.setdefault(
            result.chunk.note_id,
            [],
        ).append(result)

    for note_results in results_by_note.values():
        note = note_results[0].chunk

        with st.expander(f"{note.note_title} · {note.relative_path.as_posix()}"):
            for result in note_results:
                if result.chunk.heading_title:
                    st.caption(result.chunk.heading_title)

                st.markdown(result.chunk.text)

from collections.abc import Callable
from uuid import UUID

import streamlit as st
from tutor_bot.application.tutor_answer import TutorAnswer
from tutor_bot.application.chat_result import (
    ChatResult,
    CreateNoteDraft,
    StartRecallDraft,
    UpdateNoteDraft,
)
from tutor_bot.application.chat_service import ChatService
from tutor_bot.generation.llm_provider_error import LlmProviderError
from tutor_bot.retrieval.hybrid_search_result import HybridSearchResult
from tutor_bot.ui.speech_input import render_speech_input
from tutor_bot.ui.views.active_recall_page import open_note_study_session


_ANSWER_KEY = "tutor_answer"
_QUESTION_KEY = "chat_question"


def render_questions_page(
    answer_service_factory: Callable[[], ChatService],
) -> None:
    render_speech_input(
        _QUESTION_KEY,
        "Спросить по теме или приложению",
        "chat-speech",
    )

    with st.form("question-form"):
        question = st.text_area(
            "Спросить по теме или приложению",
            placeholder="Спросить по теме или приложению",
            label_visibility="collapsed",
            height=120,
            key=_QUESTION_KEY,
        )
        submitted = st.form_submit_button(
            "Ask",
            type="primary",
        )

    if submitted:
        _submit_question(
            answer_service_factory,
            question,
        )

    result = st.session_state.get(_ANSWER_KEY)

    if result is None:
        return

    _render_result(
        answer_service_factory,
        result,
    )


def _submit_question(
    answer_service_factory: Callable[[], ChatService],
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
        with st.spinner("Загружаю модели, ищу информацию и формирую ответ..."):
            answer_service = answer_service_factory()
            st.session_state[_ANSWER_KEY] = answer_service.answer(question)
    except LlmProviderError as error:
        st.error(f"Не удалось получить ответ от LLM: {error}")


def _render_result(
    answer_service_factory: Callable[[], ChatService],
    result: ChatResult,
) -> None:
    if result.create_note_draft is not None:
        _render_create_note_draft(
            answer_service_factory,
            result.create_note_draft,
        )

        return

    if result.update_note_draft is not None:
        _render_update_note_draft(
            answer_service_factory,
            result.update_note_draft,
        )

        return

    if result.start_recall_draft is not None:
        _render_start_recall_draft(
            answer_service_factory,
            result.start_recall_draft,
        )

        return

    if result.answer is not None:
        _render_answer(result.answer)


def _render_create_note_draft(
    answer_service_factory: Callable[[], ChatService],
    draft: CreateNoteDraft,
) -> None:
    st.divider()
    st.subheader(f"Новая заметка: {draft.title}")

    with st.expander("Сгенерированное содержание", expanded=True):
        st.markdown(draft.markdown_content)

    action_columns = st.columns(2)

    if action_columns[0].button(
        "Создать заметку",
        type="primary",
        use_container_width=True,
    ):
        try:
            created_note = answer_service_factory().create_note(draft)
        except (OSError, RuntimeError, ValueError) as error:
            st.error(f"Не удалось создать заметку: {error}")

            return

        st.session_state.pop(_ANSWER_KEY, None)
        st.success(f"Заметка «{created_note.title}» создана. ID: {created_note.id}")

    if action_columns[1].button(
        "Отменить",
        use_container_width=True,
    ):
        st.session_state.pop(_ANSWER_KEY, None)
        st.rerun()


def _render_update_note_draft(
    answer_service_factory: Callable[[], ChatService],
    draft: UpdateNoteDraft,
) -> None:
    st.divider()
    st.subheader(f"Обновление заметки: {draft.title}")

    with st.expander("Предлагаемое содержание", expanded=True):
        st.markdown(draft.markdown_content)

    action_columns = st.columns(2)

    if action_columns[0].button(
        "Обновить заметку",
        type="primary",
        use_container_width=True,
    ):
        try:
            updated_note = answer_service_factory().update_note(draft)
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            st.error(f"Не удалось обновить заметку: {error}")

            return

        st.session_state.pop(_ANSWER_KEY, None)
        st.success(f"Заметка «{updated_note.title}» обновлена.")

    if action_columns[1].button(
        "Отменить",
        use_container_width=True,
    ):
        st.session_state.pop(_ANSWER_KEY, None)
        st.rerun()


def _render_start_recall_draft(
    answer_service_factory: Callable[[], ChatService],
    draft: StartRecallDraft,
) -> None:
    st.divider()
    st.subheader(f"Active Recall: {draft.title}")
    st.write("Будет создан один вопрос по выбранной заметке.")

    if not st.button(
        "Начать Active Recall",
        type="primary",
    ):
        return

    try:
        with st.spinner("Генерирую вопрос по заметке..."):
            study_session = answer_service_factory().start_recall(draft)
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        st.error(f"Не удалось начать Active Recall: {error}")

        return

    st.session_state.pop(_ANSWER_KEY, None)
    open_note_study_session(study_session)
    st.rerun()


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

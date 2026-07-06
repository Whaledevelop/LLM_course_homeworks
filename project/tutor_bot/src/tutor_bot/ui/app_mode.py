from enum import StrEnum

import streamlit as st


APP_MODE_STATE_KEY = "selected_app_mode_v2"
_PENDING_APP_MODE_STATE_KEY = "pending_selected_app_mode"


class AppMode(StrEnum):
    QUESTIONS = "Chat"
    BROWSE_NOTES = "Просмотр и редактирование заметок"
    ADD_NOTE = "Добавление заметок"
    ASSIGNMENT_REVIEW = "Проверка заданий"
    TEST_NOTES = "Test Notes"
    PREPARE_FOR_VACANCY = "Prepare for vacancy"
    LLMS = "LLMs"
    DATABASES = "Базы данных"


def request_app_mode(app_mode: AppMode) -> None:
    st.session_state[_PENDING_APP_MODE_STATE_KEY] = app_mode


def apply_pending_app_mode() -> None:
    pending_app_mode = st.session_state.pop(_PENDING_APP_MODE_STATE_KEY, None)

    if pending_app_mode is None:
        return

    st.session_state[APP_MODE_STATE_KEY] = pending_app_mode

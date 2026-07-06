from enum import StrEnum


APP_MODE_STATE_KEY = "selected_app_mode_v2"


class AppMode(StrEnum):
    QUESTIONS = "Chat"
    BROWSE_NOTES = "Просмотр и редактирование заметок"
    ADD_NOTE = "Добавление заметок"
    ASSIGNMENT_REVIEW = "Проверка заданий"
    TEST_NOTES = "Test Notes"
    PREPARE_FOR_VACANCY = "Prepare for vacancy"
    LLMS = "LLMs"
    DATABASES = "Базы данных"

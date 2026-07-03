from enum import StrEnum


class AppMode(StrEnum):
    BROWSE_NOTES = "Просмотр и редактирование заметок"
    ADD_NOTE = "Добавление заметок"
    QUESTIONS = "Вопросы по заметкам"
    ASSIGNMENT_REVIEW = "Проверка заданий"
    TEST_NOTES = "Test Notes"
    LLMS = "LLMs"
    OBSERVABILITY = "Observability"
    DATABASES = "Базы данных"

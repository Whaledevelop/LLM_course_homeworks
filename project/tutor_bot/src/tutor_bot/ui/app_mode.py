from enum import StrEnum


class AppMode(StrEnum):
    BROWSE_NOTES = "Просмотр и редактирование заметок"
    ADD_NOTE = "Добавление заметок"
    QUESTIONS = "Вопросы по заметкам"
    ASSIGNMENT_REVIEW = "Проверка заданий"
    ACTIVE_RECALL_SESSION = "Active Recall Session"
    ACTIVE_RECALL_HISTORY = "Active Recall History"
    OBSERVABILITY = "Statistics&Observability"
    SETTINGS = "Settings"

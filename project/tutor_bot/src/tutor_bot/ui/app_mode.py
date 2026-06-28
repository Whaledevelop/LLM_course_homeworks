from enum import StrEnum


class AppMode(StrEnum):
    ADD_NOTE = "Пополнение базы знаний"
    BROWSE_NOTES = "Просмотр и редактирование"
    QUESTIONS = "Вопросы по материалам"
    ASSIGNMENT_REVIEW = "Проверка заданий"
    ACTIVE_RECALL = "Active Recall"

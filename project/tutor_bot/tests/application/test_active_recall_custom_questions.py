from unittest.mock import Mock
from uuid import uuid4

from tutor_bot.application.active_recall_service import ActiveRecallService
from tutor_bot.application.note_details import NoteDetails
from tutor_bot.application.recall_exercise import RecallExercise


def test_create_session_uses_question_from_note_metadata() -> None:
    note = NoteDetails(
        id=uuid4(),
        title="Заметка",
        group="",
        comment="",
        questions_for_tests=("Заданный вопрос?",),
        importance=5,
        knowledge=0,
        fullness=5,
        markdown_content="Содержание",
    )
    query_service = Mock()
    query_service.get_note.return_value = note
    exercise_generator = Mock()
    exercise_generator.generate.return_value = RecallExercise(
        question="Другой вопрос",
        key_points=("Первый тезис", "Второй тезис"),
        reference_answer="Ответ",
    )
    service = ActiveRecallService(
        query_service,
        exercise_generator,
        Mock(),
        Mock(),
    )

    session = service.create_session(note.id)

    exercise_generator.generate.assert_called_once_with(
        note.title,
        note.markdown_content,
        "Заданный вопрос?",
    )
    assert session.exercise.question == "Заданный вопрос?"


def test_create_session_generates_question_when_metadata_has_no_questions() -> None:
    note = NoteDetails(
        id=uuid4(),
        title="Заметка",
        group="",
        comment="",
        importance=5,
        knowledge=0,
        fullness=5,
        markdown_content="Содержание",
    )
    query_service = Mock()
    query_service.get_note.return_value = note
    exercise_generator = Mock()
    exercise_generator.generate.return_value = RecallExercise(
        question="Сгенерированный вопрос?",
        key_points=("Первый тезис", "Второй тезис"),
        reference_answer="Ответ",
    )
    service = ActiveRecallService(
        query_service,
        exercise_generator,
        Mock(),
        Mock(),
    )

    session = service.create_session(note.id)

    exercise_generator.generate.assert_called_once_with(
        note.title,
        note.markdown_content,
        None,
    )
    assert session.exercise.question == "Сгенерированный вопрос?"

from unittest.mock import Mock
from uuid import uuid4

from tutor_bot.application.active_recall_service import ActiveRecallService
from tutor_bot.application.note_details import NoteDetails
from tutor_bot.application.recall_answer_review import RecallAnswerReview
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
    service = ActiveRecallService(
        query_service,
        exercise_generator,
        Mock(),
        Mock(),
    )

    session = service.create_session(note.id)

    exercise_generator.generate.assert_not_called()
    assert session.exercise.question == "Заданный вопрос?"
    assert session.exercise.reference_answer == ""


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


def test_review_answer_generates_reference_answer_for_saved_question() -> None:
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
        question="Изменённый вопрос?",
        key_points=("Первый тезис", "Второй тезис"),
        reference_answer="Эталонный ответ",
    )
    answer_reviewer = Mock()
    answer_reviewer.review.return_value = RecallAnswerReview(
        verdict="correct",
        covered_points=(),
        missing_points=(),
        errors=(),
        feedback="Ответ принят",
    )
    service = ActiveRecallService(
        query_service,
        exercise_generator,
        answer_reviewer,
        Mock(),
    )
    session = service.create_session(note.id)

    result = service.review_answer(session, "Ответ пользователя")

    exercise_generator.generate.assert_called_once_with(
        note.title,
        note.markdown_content,
        "Заданный вопрос?",
    )
    assert result.session.exercise.question == "Заданный вопрос?"
    assert result.session.exercise.reference_answer == "Эталонный ответ"

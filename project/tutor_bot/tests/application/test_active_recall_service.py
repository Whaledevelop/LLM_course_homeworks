from uuid import UUID

from tutor_bot.application.active_recall_service import ActiveRecallService
from tutor_bot.application.delete_note_command import DeleteNoteCommand
from tutor_bot.application.note_details import NoteDetails
from tutor_bot.application.note_list_item import NoteListItem
from tutor_bot.application.recall_answer_review import RecallAnswerReview
from tutor_bot.application.recall_exercise import RecallExercise
from tutor_bot.application.recall_session_result import RecallSessionResult
from tutor_bot.application.update_note_command import UpdateNoteCommand


_FIRST_NOTE_ID = UUID("2e2a0b1a-43f0-5d43-918f-393d557d5eac")
_SECOND_NOTE_ID = UUID("00ddf68b-1408-5075-813e-9f897e8ee27a")


def test_creates_study_session_with_requested_question_count() -> None:
    service = _create_service()

    session = service.create_study_session(1)

    assert session.total_count == 1
    assert session.current_exercise.note_id in {_FIRST_NOTE_ID, _SECOND_NOTE_ID}


def test_correct_recall_answer_increases_note_knowledge() -> None:
    command_service = _FakeNoteCommandService()
    service = _create_service(command_service)
    session = service.create_study_session(1)

    service.review_study_answer(
        session,
        "student answer",
    )

    assert command_service.last_update is not None
    assert command_service.last_update.knowledge == 4


def test_clears_recall_history() -> None:
    history_writer = _FakeHistoryWriter()
    service = _create_service(history_writer=history_writer)

    service.clear_history()

    assert history_writer.was_cleared


def _create_service(
    command_service: "_FakeNoteCommandService | None" = None,
    history_writer: "_FakeHistoryWriter | None" = None,
) -> ActiveRecallService:
    return ActiveRecallService(
        _FakeNoteQueryService(),
        _FakeExerciseGenerator(),
        _FakeAnswerReviewer(),
        history_writer or _FakeHistoryWriter(),
        note_command_service=command_service,
    )


class _FakeNoteQueryService:
    def list_notes(self) -> list[NoteListItem]:
        return [
            _create_note_list_item(_FIRST_NOTE_ID),
            _create_note_list_item(_SECOND_NOTE_ID),
        ]

    def get_note(
        self,
        note_id: UUID,
    ) -> NoteDetails:
        return NoteDetails(
            id=note_id,
            title="Test note",
            group="tests",
            importance=5,
            knowledge=2,
            comment="",
            markdown_content="# Test note",
        )


class _FakeExerciseGenerator:
    def generate(
        self,
        note_title: str,
        markdown_content: str,
    ) -> RecallExercise:
        return RecallExercise(
            question=f"What is {note_title}?",
            key_points=(
                "first",
                "second",
            ),
            reference_answer=markdown_content,
        )


class _FakeAnswerReviewer:
    def review(
        self,
        exercise: RecallExercise,
        student_answer: str,
    ) -> RecallAnswerReview:
        return RecallAnswerReview(
            verdict="correct",
            covered_points=exercise.key_points,
            missing_points=(),
            errors=(),
            feedback=student_answer,
        )


class _FakeHistoryWriter:
    def __init__(self) -> None:
        self.results: list[RecallSessionResult] = []
        self.was_cleared = False

    def save(
        self,
        result: RecallSessionResult,
        review_duration_seconds: float,
    ) -> None:
        self.results.append(result)

    def load_results(self) -> tuple[RecallSessionResult, ...]:
        return tuple(self.results)

    def clear(self) -> None:
        self.results.clear()
        self.was_cleared = True


class _FakeNoteCommandService:
    def __init__(self) -> None:
        self.last_update: UpdateNoteCommand | None = None

    def update_note(
        self,
        command: UpdateNoteCommand,
    ) -> NoteDetails:
        self.last_update = command

        return NoteDetails(
            id=command.note_id,
            title=command.title,
            group=command.group,
            importance=command.importance,
            knowledge=command.knowledge,
            comment=command.comment,
            markdown_content=command.markdown_content,
        )

    def delete_note(
        self,
        command: DeleteNoteCommand,
    ) -> NoteDetails:
        raise NotImplementedError


def _create_note_list_item(
    note_id: UUID,
) -> NoteListItem:
    return NoteListItem(
        id=note_id,
        title="Test note",
        group="tests",
        importance=5,
        knowledge=2,
    )

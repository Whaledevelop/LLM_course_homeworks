from time import perf_counter
from typing import Protocol
from uuid import UUID

from tutor_bot.application.note_query_service import NoteQueryService
from tutor_bot.application.recall_session import RecallSession
from tutor_bot.application.recall_session_result import RecallSessionResult
from tutor_bot.generation.grounded_recall_answer_reviewer import (
    GroundedRecallAnswerReviewer,
)
from tutor_bot.generation.grounded_recall_exercise_generator import (
    GroundedRecallExerciseGenerator,
)


class _RecallHistoryWriter(Protocol):
    def save(
        self,
        result: RecallSessionResult,
        review_duration_seconds: float,
    ) -> None: ...


class ActiveRecallService:
    def __init__(
        self,
        note_query_service: NoteQueryService,
        exercise_generator: GroundedRecallExerciseGenerator,
        answer_reviewer: GroundedRecallAnswerReviewer,
        history_writer: _RecallHistoryWriter,
    ) -> None:
        self._note_query_service = note_query_service
        self._exercise_generator = exercise_generator
        self._answer_reviewer = answer_reviewer
        self._history_writer = history_writer

    def create_session(
        self,
        note_id: UUID,
    ) -> RecallSession:
        note = self._note_query_service.get_note(note_id)
        exercise = self._exercise_generator.generate(
            note.title,
            note.markdown_content,
        )

        return RecallSession(
            note_id=note.id,
            note_title=note.title,
            exercise=exercise,
        )

    def review_answer(
        self,
        session: RecallSession,
        student_answer: str,
    ) -> RecallSessionResult:
        normalized_answer = student_answer.strip()

        if not normalized_answer:
            raise ValueError("Student answer must not be empty")

        review_started_at = perf_counter()
        review = self._answer_reviewer.review(
            session.exercise,
            normalized_answer,
        )
        review_duration_seconds = perf_counter() - review_started_at
        result = RecallSessionResult(
            session=session,
            student_answer=normalized_answer,
            review=review,
        )
        self._history_writer.save(
            result,
            review_duration_seconds,
        )

        return result

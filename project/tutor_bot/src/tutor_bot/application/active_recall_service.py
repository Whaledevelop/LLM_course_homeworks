from random import Random, SystemRandom
from time import perf_counter
from typing import Protocol
from uuid import UUID

from tutor_bot.application.note_query_service import NoteQueryService
from tutor_bot.application.recall_session import RecallSession
from tutor_bot.application.recall_session_result import RecallSessionResult
from tutor_bot.application.recall_study_session import RecallStudySession
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

    def load_results(self) -> tuple[RecallSessionResult, ...]: ...


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
            source_markdown=note.markdown_content,
            exercise=exercise,
        )

    def create_study_session(
        self,
        first_note_id: UUID,
    ) -> RecallStudySession:
        notes = self._note_query_service.list_notes()
        remaining_note_ids = [note.id for note in notes if note.id != first_note_id]
        seed = SystemRandom().randrange(2**32)
        random_generator = Random(seed)
        selected_note_ids = random_generator.sample(
            remaining_note_ids,
            k=min(9, len(remaining_note_ids)),
        )
        note_ids = (first_note_id, *selected_note_ids)

        return RecallStudySession(
            seed=seed,
            note_ids=note_ids,
            current_index=0,
            current_exercise=self.create_session(first_note_id),
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

    def get_history(self) -> tuple[RecallSessionResult, ...]:
        return self._history_writer.load_results()

    def review_study_answer(
        self,
        study_session: RecallStudySession,
        student_answer: str,
    ) -> RecallStudySession:
        if study_session.answered_count != study_session.current_index:
            raise ValueError("Current exercise has already been reviewed")

        result = self.review_answer(
            study_session.current_exercise,
            student_answer,
        )

        return study_session.model_copy(
            update={"results": (*study_session.results, result)},
        )

    def advance_study_session(
        self,
        study_session: RecallStudySession,
    ) -> RecallStudySession:
        if study_session.answered_count <= study_session.current_index:
            raise ValueError("Current exercise must be reviewed before advancing")

        next_index = study_session.current_index + 1

        if next_index >= study_session.total_count:
            return study_session

        next_exercise = self.create_session(study_session.note_ids[next_index])

        return study_session.model_copy(
            update={
                "current_index": next_index,
                "current_exercise": next_exercise,
            },
        )

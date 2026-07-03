from random import Random, SystemRandom
from time import perf_counter
from typing import Protocol
from uuid import UUID, uuid4

from tutor_bot.application.note_query_service import NoteQueryService
from tutor_bot.application.note_command_service import NoteCommandService
from tutor_bot.application.recall_session import RecallSession
from tutor_bot.application.recall_session_result import RecallSessionResult
from tutor_bot.application.recall_study_session import RecallStudySession
from tutor_bot.application.update_note_command import UpdateNoteCommand
from tutor_bot.generation.grounded_recall_answer_reviewer import (
    GroundedRecallAnswerReviewer,
)
from tutor_bot.generation.grounded_recall_exercise_generator import (
    GroundedRecallExerciseGenerator,
)
from tutor_bot.schemas.observability_event import ObservabilityEvent


class _RecallHistoryWriter(Protocol):
    def save(
        self,
        result: RecallSessionResult,
        review_duration_seconds: float,
    ) -> None: ...

    def load_results(self) -> tuple[RecallSessionResult, ...]: ...

    def clear(self) -> None: ...


class _ObservabilityEventRecorder(Protocol):
    def record(
        self,
        event: ObservabilityEvent,
    ) -> None:
        pass


class ActiveRecallService:
    def __init__(
        self,
        note_query_service: NoteQueryService,
        exercise_generator: GroundedRecallExerciseGenerator,
        answer_reviewer: GroundedRecallAnswerReviewer,
        history_writer: _RecallHistoryWriter,
        note_command_service: NoteCommandService | None = None,
        observability_event_service: _ObservabilityEventRecorder | None = None,
    ) -> None:
        self._note_query_service = note_query_service
        self._exercise_generator = exercise_generator
        self._answer_reviewer = answer_reviewer
        self._history_writer = history_writer
        self._note_command_service = note_command_service
        self._observability_event_service = observability_event_service

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
        question_count: int,
    ) -> RecallStudySession:
        if question_count <= 0:
            raise ValueError("Question count must be positive")

        notes = self._note_query_service.list_notes()
        note_ids = [note.id for note in notes]

        if not note_ids:
            raise ValueError("Active Recall requires at least one note")

        seed = SystemRandom().randrange(2**32)
        random_generator = Random(seed)
        selected_note_ids = tuple(
            random_generator.sample(
                note_ids,
                k=min(question_count, len(note_ids)),
            )
        )

        return RecallStudySession(
            seed=seed,
            note_ids=selected_note_ids,
            current_index=0,
            current_exercise=self.create_session(selected_note_ids[0]),
        )

    def review_answer(
        self,
        session: RecallSession,
        student_answer: str,
    ) -> RecallSessionResult:
        normalized_answer = student_answer.strip()

        if not normalized_answer:
            raise ValueError("Student answer must not be empty")

        trace_id = str(uuid4())
        review_started_at = perf_counter()
        self._record_event(
            ObservabilityEvent(
                scenario="active_recall",
                event_type="answer_review",
                status="started",
                trace_id=trace_id,
                session_id=str(session.note_id),
                payload={
                    "note_id": str(session.note_id),
                    "note_title": session.note_title,
                    "student_answer_length": len(normalized_answer),
                    "key_points_count": len(session.exercise.key_points),
                },
            )
        )

        try:
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
            self._record_event(
                ObservabilityEvent(
                    scenario="active_recall",
                    event_type="answer_review",
                    status="succeeded",
                    trace_id=trace_id,
                    session_id=str(session.note_id),
                    duration_seconds=review_duration_seconds,
                    payload={
                        "note_id": str(session.note_id),
                        "note_title": session.note_title,
                        "student_answer_length": len(normalized_answer),
                        "key_points_count": len(session.exercise.key_points),
                        "verdict": review.verdict,
                        "covered_points_count": len(review.covered_points),
                        "missing_points_count": len(review.missing_points),
                        "errors_count": len(review.errors),
                    },
                )
            )
        except Exception as exception:
            self._record_event(
                ObservabilityEvent(
                    scenario="active_recall",
                    event_type="answer_review",
                    status="failed",
                    trace_id=trace_id,
                    session_id=str(session.note_id),
                    duration_seconds=perf_counter() - review_started_at,
                    payload={
                        "note_id": str(session.note_id),
                        "note_title": session.note_title,
                        "student_answer_length": len(normalized_answer),
                        "key_points_count": len(session.exercise.key_points),
                    },
                    error=exception.__class__.__name__,
                )
            )
            raise

        return result

    def get_history(self) -> tuple[RecallSessionResult, ...]:
        return self._history_writer.load_results()

    def clear_history(self) -> None:
        self._history_writer.clear()

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
        self._increase_note_knowledge(result)

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

    def _record_event(
        self,
        event: ObservabilityEvent,
    ) -> None:
        if self._observability_event_service is None:
            return

        self._observability_event_service.record(event)

    def _increase_note_knowledge(
        self,
        result: RecallSessionResult,
    ) -> None:
        if self._note_command_service is None:
            return

        increment = 0

        if result.review.verdict == "correct":
            increment = 2
        elif result.review.verdict == "partially_correct":
            increment = 1

        if increment == 0:
            return

        note = self._note_query_service.get_note(result.session.note_id)
        updated_knowledge = min(10, note.knowledge + increment)

        if updated_knowledge == note.knowledge:
            return

        self._note_command_service.update_note(
            UpdateNoteCommand(
                note_id=note.id,
                title=note.title,
                group=note.group,
                comment=note.comment,
                importance=note.importance,
                knowledge=updated_knowledge,
                markdown_content=note.markdown_content,
            )
        )

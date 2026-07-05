from uuid import uuid4

from tutor_bot.application.recall_answer_review import RecallAnswerReview
from tutor_bot.application.recall_exercise import RecallExercise
from tutor_bot.application.recall_session import RecallSession
from tutor_bot.application.recall_session_result import RecallSessionResult


def test_old_recall_history_record_remains_compatible() -> None:
    result = RecallSessionResult.model_validate(
        {
            "session": {
                "note_id": str(uuid4()),
                "note_title": "ECS",
                "source_markdown": "Entities and systems",
                "exercise": RecallExercise(
                    question="What is ECS?",
                    key_points=("Entities", "Systems"),
                    reference_answer="Entities hold data and systems process it.",
                ).model_dump(mode="json"),
            },
            "student_answer": "Architecture pattern",
            "review": RecallAnswerReview(
                verdict="partially_correct",
                covered_points=("Entities",),
                missing_points=("Systems",),
                errors=(),
                feedback="Explain systems.",
            ).model_dump(mode="json"),
        }
    )

    assert isinstance(result.session, RecallSession)
    assert result.session.context_title is None
    assert result.session.focus_topic is None

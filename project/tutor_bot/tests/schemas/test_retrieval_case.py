import pytest
from pydantic import ValidationError

from tutor_bot.schemas.retrieval_case import RetrievalCase


def test_accepts_answerable_case() -> None:
    retrieval_case = RetrievalCase(
        id="threads-parallelism",
        question="Чем параллелизм отличается от многопоточности?",
        expected_note_ids=["e248dd6f-963d-552f-8616-d87082905b4e"],
    )

    assert retrieval_case.should_find_answer is True


def test_accepts_unanswerable_case() -> None:
    retrieval_case = RetrievalCase(
        id="missing-information",
        question="Как настроить Kubernetes-кластер?",
        should_find_answer=False,
    )

    assert retrieval_case.expected_note_ids == []


def test_rejects_answerable_case_without_expected_notes() -> None:
    with pytest.raises(ValidationError):
        RetrievalCase(
            id="invalid-case",
            question="Что такое сборщик мусора?",
        )
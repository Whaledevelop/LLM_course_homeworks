from pathlib import Path

import pytest

from tutor_bot.application.observability_event_service import ObservabilityEventService
from tutor_bot.infrastructure.jsonl_observability_event_repository import (
    JsonlObservabilityEventRepository,
)
from tutor_bot.schemas.observability_event import ObservabilityEvent


def test_returns_empty_events_when_file_is_missing(
    tmp_path: Path,
) -> None:
    repository = JsonlObservabilityEventRepository(tmp_path / "events.jsonl")

    assert repository.load_events() == ()


def test_appends_and_loads_events(
    tmp_path: Path,
) -> None:
    events_file = tmp_path / "history" / "events.jsonl"
    repository = JsonlObservabilityEventRepository(events_file)
    event = ObservabilityEvent(
        scenario="rag_answer",
        event_type="pipeline",
        status="succeeded",
        duration_seconds=1.25,
        payload={
            "candidate_count": 3,
        },
    )

    repository.append(event)

    assert repository.load_events() == (event,)
    assert events_file.read_text(encoding="utf-8").count("\n") == 1


def test_builds_observability_statistics(
    tmp_path: Path,
) -> None:
    service = ObservabilityEventService(
        JsonlObservabilityEventRepository(tmp_path / "events.jsonl")
    )
    service.record(
        ObservabilityEvent(
            scenario="rag_answer",
            event_type="pipeline",
            status="succeeded",
            duration_seconds=1.0,
        )
    )
    service.record(
        ObservabilityEvent(
            scenario="rag_answer",
            event_type="pipeline",
            status="succeeded",
            duration_seconds=2.0,
        )
    )
    service.record(
        ObservabilityEvent(
            scenario="active_recall",
            event_type="answer_review",
            status="failed",
            duration_seconds=0.5,
            error="RuntimeError",
        )
    )

    statistics = service.get_statistics()

    assert statistics.total_events == 3
    assert statistics.events_by_scenario == {
        "active_recall": 1,
        "rag_answer": 2,
    }
    assert statistics.events_by_status == {
        "failed": 1,
        "succeeded": 2,
    }
    assert statistics.average_duration_seconds_by_scenario == {
        "active_recall": 0.5,
        "rag_answer": 1.5,
    }
    assert len(statistics.latest_errors) == 1
    assert statistics.latest_errors[0].error == "RuntimeError"


def test_rejects_invalid_latest_error_limit(
    tmp_path: Path,
) -> None:
    service = ObservabilityEventService(
        JsonlObservabilityEventRepository(tmp_path / "events.jsonl")
    )

    with pytest.raises(ValueError, match="Latest error limit must be positive"):
        service.get_statistics(latest_error_limit=0)

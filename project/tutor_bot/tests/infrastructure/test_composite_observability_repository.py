from pathlib import Path

from tutor_bot.application.observability_event_service import (
    ObservabilityEventService,
    add_current_observation_metadata,
)
from tutor_bot.infrastructure.composite_observability_event_repository import (
    CompositeObservabilityEventRepository,
)
from tutor_bot.infrastructure.jsonl_observability_event_repository import (
    JsonlObservabilityEventRepository,
)
from tutor_bot.schemas.observability_event import ObservabilityEvent


class _UnavailableRepository:
    def append(self, event: ObservabilityEvent) -> None:
        raise ConnectionError("Unavailable")


def test_keeps_local_event_when_optional_sink_is_unavailable(tmp_path: Path) -> None:
    local_repository = JsonlObservabilityEventRepository(tmp_path / "events.jsonl")
    repository = CompositeObservabilityEventRepository(
        local_repository,
        (_UnavailableRepository(),),
        langfuse_enabled=True,
    )
    event = ObservabilityEvent(
        scenario="rag_answer",
        event_type="pipeline",
        observation_type="trace",
        status="succeeded",
    )

    repository.append(event)

    assert local_repository.load_events() == (event,)


def test_observation_scope_preserves_trace_hierarchy_and_usage(tmp_path: Path) -> None:
    repository = JsonlObservabilityEventRepository(tmp_path / "events.jsonl")
    service = ObservabilityEventService(repository)

    with service.observe("rag_answer", "pipeline", observation_type="trace") as trace:
        with service.observe("rag_answer", "generation", observation_type="generation"):
            add_current_observation_metadata(
                provider="ollama",
                model="qwen3.5:9b",
                total_tokens=42,
            )

    events = repository.load_events()
    generation_event = next(
        event
        for event in events
        if event.event_type == "generation" and event.status == "succeeded"
    )

    assert generation_event.trace_id == trace.trace_id
    assert generation_event.parent_observation_id == trace.observation_id
    assert generation_event.payload["total_tokens"] == 42

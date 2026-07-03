from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from typing import Any, Iterator, Protocol
from uuid import uuid4

from tutor_bot.application.observability_scope import ObservabilityScope
from tutor_bot.application.observability_sink_status import ObservabilitySinkStatus
from tutor_bot.application.observability_statistics import ObservabilityStatistics
from tutor_bot.schemas.observability_event import ObservabilityEvent


class _ObservabilityEventRepository(Protocol):
    def append(
        self,
        event: ObservabilityEvent,
    ) -> None:
        pass

    def load_events(
        self,
    ) -> tuple[ObservabilityEvent, ...]:
        pass


_current_scope: ContextVar[ObservabilityScope | None] = ContextVar(
    "current_observability_scope",
    default=None,
)


class ObservabilityEventService:
    def __init__(
        self,
        event_repository: _ObservabilityEventRepository,
    ) -> None:
        self._event_repository = event_repository

    def record(
        self,
        event: ObservabilityEvent,
    ) -> None:
        self._event_repository.append(event)

    @contextmanager
    def observe(
        self,
        scenario: str,
        name: str,
        observation_type: str = "span",
        payload: dict[str, Any] | None = None,
        session_id: str | None = None,
        trace_id: str | None = None,
        parent_observation_id=None,
    ) -> Iterator[ObservabilityScope]:
        parent_scope = _current_scope.get()
        resolved_trace_id = trace_id or (
            parent_scope.trace_id if parent_scope is not None else str(uuid4())
        )
        scope = ObservabilityScope(
            trace_id=resolved_trace_id,
            observation_id=uuid4(),
            payload=dict(payload or {}),
        )
        started_at = perf_counter()
        event_arguments = {
            "scenario": scenario,
            "event_type": name,
            "observation_type": observation_type,
            "observation_id": scope.observation_id,
            "parent_observation_id": parent_observation_id
            or (parent_scope.observation_id if parent_scope is not None else None),
            "trace_id": resolved_trace_id,
            "session_id": session_id,
        }
        self.record(
            ObservabilityEvent(
                **event_arguments,
                status="started",
                payload=scope.payload,
            )
        )
        context_token = _current_scope.set(scope)

        try:
            yield scope
        except Exception as exception:
            self.record(
                ObservabilityEvent(
                    **event_arguments,
                    status="failed",
                    duration_seconds=perf_counter() - started_at,
                    payload=scope.payload,
                    error=exception.__class__.__name__,
                )
            )
            raise
        else:
            self.record(
                ObservabilityEvent(
                    **event_arguments,
                    status="succeeded",
                    duration_seconds=perf_counter() - started_at,
                    payload=scope.payload,
                )
            )
        finally:
            _current_scope.reset(context_token)

    def add_current_metadata(self, **metadata: Any) -> None:
        scope = _current_scope.get()

        if scope is not None:
            scope.add_metadata(**metadata)

    def get_sink_statuses(self) -> tuple[ObservabilitySinkStatus, ...]:
        get_statuses = getattr(self._event_repository, "get_statuses", None)

        if get_statuses is None:
            return (
                ObservabilitySinkStatus(
                    name="Local JSONL",
                    enabled=True,
                    available=True,
                    message="Активен",
                ),
            )

        return get_statuses()

    def record_feedback(
        self,
        trace_id: str,
        score: float,
        comment: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self.record(
            ObservabilityEvent(
                scenario="user_feedback",
                event_type="user_feedback",
                observation_type="evaluator",
                status="succeeded",
                trace_id=trace_id,
                session_id=session_id,
                payload={"score": score, "comment": comment},
            )
        )

    def load_events(
        self,
    ) -> tuple[ObservabilityEvent, ...]:
        return self._event_repository.load_events()

    def get_statistics(
        self,
        latest_error_limit: int = 5,
    ) -> ObservabilityStatistics:
        if latest_error_limit <= 0:
            raise ValueError("Latest error limit must be positive")

        events = self.load_events()
        events_by_scenario: dict[str, int] = {}
        events_by_event_type: dict[str, int] = {}
        events_by_observation_type: dict[str, int] = {}
        events_by_status: dict[str, int] = {}
        events_by_model: dict[str, int] = {}
        terminal_statuses_by_scenario: dict[str, dict[str, int]] = {}
        durations_by_scenario: dict[str, list[float]] = {}
        failed_events = []

        for event in events:
            events_by_scenario[event.scenario] = events_by_scenario.get(event.scenario, 0) + 1
            events_by_event_type[event.event_type] = (
                events_by_event_type.get(event.event_type, 0) + 1
            )
            events_by_observation_type[event.observation_type] = (
                events_by_observation_type.get(event.observation_type, 0) + 1
            )
            events_by_status[event.status] = events_by_status.get(event.status, 0) + 1

            provider = event.payload.get("provider")
            model = event.payload.get("model") or event.payload.get("model_name")

            if isinstance(provider, str) and isinstance(model, str):
                model_key = f"{provider} / {model}"
                events_by_model[model_key] = events_by_model.get(model_key, 0) + 1

            if event.status in {"succeeded", "failed"}:
                scenario_statuses = terminal_statuses_by_scenario.setdefault(
                    event.scenario,
                    {"succeeded": 0, "failed": 0},
                )
                scenario_statuses[event.status] += 1

            if event.duration_seconds is not None:
                durations_by_scenario.setdefault(
                    event.scenario,
                    [],
                ).append(event.duration_seconds)

            if event.status == "failed":
                failed_events.append(event)

        average_duration_seconds_by_scenario = {
            scenario: round(sum(durations) / len(durations), 3)
            for scenario, durations in durations_by_scenario.items()
        }
        success_rate_by_scenario = {
            scenario: round(
                statuses["succeeded"] / (statuses["succeeded"] + statuses["failed"]) * 100,
                1,
            )
            for scenario, statuses in terminal_statuses_by_scenario.items()
        }
        latest_errors = tuple(
            sorted(
                failed_events,
                key=lambda event: event.recorded_at,
                reverse=True,
            )[:latest_error_limit]
        )

        return ObservabilityStatistics(
            total_events=len(events),
            events_by_scenario=events_by_scenario,
            events_by_event_type=events_by_event_type,
            events_by_observation_type=events_by_observation_type,
            events_by_status=events_by_status,
            events_by_model=events_by_model,
            success_rate_by_scenario=success_rate_by_scenario,
            average_duration_seconds_by_scenario=average_duration_seconds_by_scenario,
            latest_errors=latest_errors,
        )


def add_current_observation_metadata(**metadata: Any) -> None:
    scope = _current_scope.get()

    if scope is not None:
        scope.add_metadata(**metadata)

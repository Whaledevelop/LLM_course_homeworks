from typing import Protocol

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
                statuses["succeeded"]
                / (statuses["succeeded"] + statuses["failed"])
                * 100,
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
            events_by_status=events_by_status,
            events_by_model=events_by_model,
            success_rate_by_scenario=success_rate_by_scenario,
            average_duration_seconds_by_scenario=average_duration_seconds_by_scenario,
            latest_errors=latest_errors,
        )

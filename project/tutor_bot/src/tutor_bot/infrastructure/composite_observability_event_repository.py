from tutor_bot.application.observability_sink_status import ObservabilitySinkStatus
from tutor_bot.schemas.observability_event import ObservabilityEvent


class CompositeObservabilityEventRepository:
    def __init__(
        self,
        primary_repository,
        optional_repositories: tuple = (),
        langfuse_enabled: bool = False,
    ) -> None:
        self._primary_repository = primary_repository
        self._optional_repositories = optional_repositories
        self._langfuse_enabled = langfuse_enabled

    def append(self, event: ObservabilityEvent) -> None:
        self._primary_repository.append(event)

        for repository in self._optional_repositories:
            try:
                repository.append(event)
            except Exception:
                continue

    def load_events(self) -> tuple[ObservabilityEvent, ...]:
        return self._primary_repository.load_events()

    def get_statuses(self) -> tuple[ObservabilitySinkStatus, ...]:
        statuses = [
            ObservabilitySinkStatus(
                name="Local JSONL",
                enabled=True,
                available=True,
                message="Активен",
            )
        ]

        if not self._langfuse_enabled:
            statuses.append(
                ObservabilitySinkStatus(
                    name="Langfuse",
                    enabled=False,
                    message="Выключен через LANGFUSE_ENABLED",
                )
            )

        for repository in self._optional_repositories:
            statuses.append(repository.get_status())

        return tuple(statuses)

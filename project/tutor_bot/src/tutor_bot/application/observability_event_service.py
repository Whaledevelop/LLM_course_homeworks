from typing import Protocol

from tutor_bot.schemas.observability_event import ObservabilityEvent


class _ObservabilityEventWriter(Protocol):
    def append(
        self,
        event: ObservabilityEvent,
    ) -> None:
        pass


class ObservabilityEventService:
    def __init__(
        self,
        event_writer: _ObservabilityEventWriter,
    ) -> None:
        self._event_writer = event_writer

    def record(
        self,
        event: ObservabilityEvent,
    ) -> None:
        self._event_writer.append(event)

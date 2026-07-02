from pathlib import Path

from tutor_bot.schemas.observability_event import ObservabilityEvent


class JsonlObservabilityEventRepository:
    def __init__(
        self,
        events_file: Path,
    ) -> None:
        self._events_file = events_file

    def append(
        self,
        event: ObservabilityEvent,
    ) -> None:
        self._events_file.parent.mkdir(parents=True, exist_ok=True)
        serialized_event = event.model_dump_json()

        with self._events_file.open(
            "a",
            encoding="utf-8",
            newline="\n",
        ) as events_file:
            events_file.write(serialized_event + "\n")

    def load_events(
        self,
    ) -> tuple[ObservabilityEvent, ...]:
        if not self._events_file.exists():
            return ()

        events = []

        for line in self._events_file.read_text(encoding="utf-8").splitlines():
            events.append(ObservabilityEvent.model_validate_json(line))

        return tuple(events)

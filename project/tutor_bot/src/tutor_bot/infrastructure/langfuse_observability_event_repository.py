from typing import Any

from langfuse import Langfuse

from tutor_bot.application.observability_sink_status import ObservabilitySinkStatus
from tutor_bot.schemas.observability_event import ObservabilityEvent


class LangfuseObservabilityEventRepository:
    def __init__(
        self,
        public_key: str,
        secret_key: str,
        base_url: str,
    ) -> None:
        self._client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=base_url,
        )
        self._observations: dict[str, Any] = {}

    def append(self, event: ObservabilityEvent) -> None:
        observation_key = str(event.observation_id)

        if event.status == "started":
            self._observations[observation_key] = self._start_observation(event)

            return

        observation = self._observations.pop(observation_key, None)

        if observation is None:
            observation = self._start_observation(event)

        payload = dict(event.payload)
        payload["duration_seconds"] = event.duration_seconds
        payload["status"] = event.status
        output = payload.pop("output", None)
        model = payload.get("model") or payload.get("model_name")
        usage_details = self._build_usage_details(payload)
        observation.update(
            output=output,
            metadata=payload,
            model=model if isinstance(model, str) else None,
            usage_details=usage_details or None,
            level="ERROR" if event.status == "failed" else "DEFAULT",
            status_message=event.error,
        )
        observation.end()

    def check_connection(self) -> bool:
        return self._client.auth_check()

    def get_status(self) -> ObservabilitySinkStatus:
        try:
            is_available = self.check_connection()
        except Exception as exception:
            return ObservabilitySinkStatus(
                name="Langfuse",
                enabled=True,
                available=False,
                message=exception.__class__.__name__,
            )

        return ObservabilitySinkStatus(
            name="Langfuse",
            enabled=True,
            available=is_available,
            message="Подключён" if is_available else "Недоступен",
        )

    def flush(self) -> None:
        self._client.flush()

    def _start_observation(self, event: ObservabilityEvent):
        trace_id = self._client.create_trace_id(seed=event.trace_id or str(event.event_id))
        trace_context = {"trace_id": trace_id}
        parent = self._observations.get(str(event.parent_observation_id))

        if parent is not None:
            trace_context = {
                "trace_id": parent.trace_id,
                "parent_span_id": parent.id,
            }

        observation_type = (
            "span" if event.observation_type in {"trace", "event"} else event.observation_type
        )

        return self._client.start_observation(
            trace_context=trace_context,
            name=event.event_type,
            as_type=observation_type,
            input=event.payload.get("input"),
            metadata={
                **event.payload,
                "scenario": event.scenario,
                "session_id": event.session_id,
                "local_observation_id": str(event.observation_id),
            },
        )

    def _build_usage_details(self, payload: dict[str, Any]) -> dict[str, int]:
        usage_details = {}

        for source_name, target_name in (
            ("prompt_tokens", "input"),
            ("completion_tokens", "output"),
            ("total_tokens", "total"),
        ):
            value = payload.get(source_name)

            if isinstance(value, int):
                usage_details[target_name] = value

        return usage_details

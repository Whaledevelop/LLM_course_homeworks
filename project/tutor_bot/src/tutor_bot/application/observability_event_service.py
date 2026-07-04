from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from typing import Any, Iterator, Protocol
from uuid import uuid4

from tutor_bot.application.observability_scope import ObservabilityScope
from tutor_bot.schemas.observability_event import ObservabilityEvent


class _ObservabilityEventRepository(Protocol):
    def append(
        self,
        event: ObservabilityEvent,
    ) -> None:
        pass


_current_scope: ContextVar[ObservabilityScope | None] = ContextVar(
    "current_observability_scope",
    default=None,
)


class ObservabilityEventService:
    def __init__(
        self,
        event_repository: _ObservabilityEventRepository,
        generation_provider: str | None = None,
        generation_model: str | None = None,
    ) -> None:
        self._event_repository = event_repository
        self._generation_provider = generation_provider
        self._generation_model = generation_model

    def record(
        self,
        event: ObservabilityEvent,
    ) -> None:
        if event.observation_type == "generation":
            payload = dict(event.payload)
            payload.setdefault("provider", self._generation_provider)
            payload.setdefault("model", self._generation_model)

            if event.status in {"succeeded", "failed"}:
                payload.setdefault("prompt_tokens", 0)
                payload.setdefault("completion_tokens", 0)
                payload.setdefault("total_tokens", 0)

            if event.status == "failed":
                payload.setdefault("error", event.error)

            event = event.model_copy(update={"payload": payload})

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
            observation_type=observation_type,
            payload=dict(payload or {}),
        )

        if observation_type == "generation":
            scope.payload.setdefault("provider", self._generation_provider)
            scope.payload.setdefault("model", self._generation_model)
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


def add_current_observation_metadata(**metadata: Any) -> None:
    scope = _current_scope.get()

    if scope is not None:
        scope.add_metadata(**metadata)


def is_generation_observation_active() -> bool:
    scope = _current_scope.get()

    return scope is not None and scope.observation_type == "generation"

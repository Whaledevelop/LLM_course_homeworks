from datetime import datetime, timezone
from typing import Literal
from uuid import UUID, uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

type JsonValue = str | int | float | bool | None | tuple["JsonValue", ...] | dict[str, "JsonValue"]


class ObservabilityEvent(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    event_id: UUID = Field(default_factory=uuid4)
    recorded_at: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    scenario: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    observation_type: Literal[
        "trace",
        "span",
        "generation",
        "retriever",
        "evaluator",
        "event",
    ] = "event"
    observation_id: UUID = Field(default_factory=uuid4)
    parent_observation_id: UUID | None = None
    status: Literal["started", "succeeded", "failed", "skipped"]
    trace_id: str | None = Field(default=None, min_length=1)
    session_id: str | None = Field(default=None, min_length=1)
    duration_seconds: float | None = Field(default=None, ge=0)
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    error: str | None = Field(default=None, min_length=1)

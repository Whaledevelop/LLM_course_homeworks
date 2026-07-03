from pydantic import BaseModel, ConfigDict, Field

from tutor_bot.schemas.observability_event import ObservabilityEvent


class ObservabilityStatistics(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    total_events: int = Field(ge=0)
    events_by_scenario: dict[str, int]
    events_by_event_type: dict[str, int]
    events_by_observation_type: dict[str, int]
    events_by_status: dict[str, int]
    events_by_model: dict[str, int]
    success_rate_by_scenario: dict[str, float]
    average_duration_seconds_by_scenario: dict[str, float]
    latest_errors: tuple[ObservabilityEvent, ...]

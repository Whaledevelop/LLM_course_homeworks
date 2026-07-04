from tutor_bot.infrastructure.langfuse_observability_event_repository import (
    LangfuseObservabilityEventRepository,
)
from tutor_bot.schemas.observability_event import ObservabilityEvent


class _Observation:
    trace_id = "trace-id"
    id = "observation-id"

    def __init__(self) -> None:
        self.update_arguments = {}
        self.ended = False

    def update(self, **kwargs) -> None:
        self.update_arguments = kwargs

    def end(self) -> None:
        self.ended = True


class _Client:
    def __init__(self) -> None:
        self.start_arguments = {}
        self.observation = _Observation()

    def create_trace_id(self, seed: str) -> str:
        return "trace-id"

    def start_observation(self, **kwargs):
        self.start_arguments = kwargs

        return self.observation


def test_generation_mapper_sets_model_on_start_and_usage_on_end() -> None:
    repository = LangfuseObservabilityEventRepository.__new__(LangfuseObservabilityEventRepository)
    repository._client = _Client()
    repository._observations = {}
    observation_id = ObservabilityEvent(
        scenario="rag_answer",
        event_type="generation",
        observation_type="generation",
        status="started",
    ).observation_id
    repository.append(
        ObservabilityEvent(
            scenario="rag_answer",
            event_type="generation",
            observation_type="generation",
            observation_id=observation_id,
            status="started",
            payload={"model": "qwen3.5:9b", "provider": "ollama", "input": "question"},
        )
    )
    repository.append(
        ObservabilityEvent(
            scenario="rag_answer",
            event_type="generation",
            observation_type="generation",
            observation_id=observation_id,
            status="succeeded",
            duration_seconds=1.5,
            payload={
                "model": "qwen3.5:9b",
                "provider": "ollama",
                "output": "answer",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        )
    )

    assert repository._client.start_arguments["as_type"] == "generation"
    assert repository._client.start_arguments["model"] == "qwen3.5:9b"
    assert repository._client.observation.update_arguments["output"] == "answer"
    assert repository._client.observation.update_arguments["usage_details"] == {
        "input": 10,
        "output": 5,
        "total": 15,
    }
    assert repository._client.observation.ended is True

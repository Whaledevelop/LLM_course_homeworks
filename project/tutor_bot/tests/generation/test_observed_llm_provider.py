from tutor_bot.application.observability_event_service import ObservabilityEventService
from tutor_bot.generation.llm_response import LlmResponse
from tutor_bot.generation.observed_llm_provider import ObservedLlmProvider
from tutor_bot.schemas.observability_event import ObservabilityEvent


class _EventRepository:
    def __init__(self) -> None:
        self.events: list[ObservabilityEvent] = []

    def append(self, event: ObservabilityEvent) -> None:
        self.events.append(event)


class _Provider:
    provider_name = "ollama"
    model_name = "qwen3.5:9b"

    def generate(self, **kwargs) -> LlmResponse:
        return LlmResponse(
            text="answer",
            provider=self.provider_name,
            model=self.model_name,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )


def test_standalone_generation_uses_observation_service_mapper() -> None:
    repository = _EventRepository()
    service = ObservabilityEventService(
        repository,
        generation_provider="ollama",
        generation_model="qwen3.5:9b",
    )
    provider = ObservedLlmProvider(_Provider(), service)

    provider.generate(
        messages=[{"role": "user", "content": "question"}],
        temperature=0.1,
        max_tokens=100,
    )

    started, succeeded = repository.events
    assert started.observation_type == "generation"
    assert started.payload["model"] == "qwen3.5:9b"
    assert succeeded.payload["output"] == "answer"
    assert succeeded.payload["total_tokens"] == 15
    assert succeeded.duration_seconds is not None


def test_pipeline_generation_is_not_duplicated() -> None:
    repository = _EventRepository()
    service = ObservabilityEventService(
        repository,
        generation_provider="ollama",
        generation_model="qwen3.5:9b",
    )
    provider = ObservedLlmProvider(_Provider(), service)

    with service.observe(
        "rag_answer",
        "generation",
        observation_type="generation",
        payload={"input": {"question": "question"}},
    ) as scope:
        response = provider.generate(
            messages=[{"role": "user", "content": "question"}],
            temperature=0.1,
            max_tokens=100,
        )
        scope.set_output(response.text)

    assert len(repository.events) == 2
    assert repository.events[0].payload["model"] == "qwen3.5:9b"
    assert repository.events[1].payload["total_tokens"] == 15


def test_failed_generation_keeps_identity_and_zero_usage() -> None:
    class _FailingProvider(_Provider):
        def generate(self, **kwargs) -> LlmResponse:
            raise RuntimeError("failed")

    repository = _EventRepository()
    service = ObservabilityEventService(
        repository,
        generation_provider="ollama",
        generation_model="qwen3.5:9b",
    )
    provider = ObservedLlmProvider(_FailingProvider(), service)

    try:
        provider.generate(
            messages=[{"role": "user", "content": "question"}],
            temperature=0.1,
            max_tokens=100,
        )
    except RuntimeError:
        pass

    failed = repository.events[-1]
    assert failed.status == "failed"
    assert failed.payload["model"] == "qwen3.5:9b"
    assert failed.payload["error"] == "RuntimeError"
    assert failed.payload["total_tokens"] == 0

from tutor_bot.generation.llm_response import LlmResponse
from tutor_bot.infrastructure.llm_usage_repository import LlmUsageRepository


def test_persists_usage_between_repository_instances(tmp_path) -> None:
    usage_file = tmp_path / "history" / "llm_usage.jsonl"
    response = LlmResponse(
        text="answer",
        provider="ollama",
        model="test-model",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )

    LlmUsageRepository(usage_file).save(response)

    persisted_usage = LlmUsageRepository(usage_file).load()

    assert persisted_usage == (response,)

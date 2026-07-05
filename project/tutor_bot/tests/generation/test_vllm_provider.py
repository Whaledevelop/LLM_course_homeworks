from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from tutor_bot.generation.llm_provider_error import LlmProviderError
from tutor_bot.generation.vllm_provider import VllmProvider


def test_generate_returns_response_and_records_usage() -> None:
    usage_callback = Mock()
    provider = VllmProvider(
        "http://localhost:8000/v1",
        "secret",
        "Qwen/Qwen3-8B",
        usage_callback=usage_callback,
    )
    provider._client = _create_client(content="answer")

    result = provider.generate(
        [{"role": "user", "content": "question"}],
        temperature=0.2,
        max_tokens=256,
    )

    assert result.text == "answer"
    assert result.provider == "vllm"
    assert result.model == "Qwen/Qwen3-8B"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 4
    assert result.total_tokens == 14
    usage_callback.assert_called_once_with(result)


def test_generate_passes_json_schema_to_vllm() -> None:
    provider = VllmProvider(
        "http://localhost:8000/v1",
        "secret",
        "Qwen/Qwen3-8B",
    )
    provider._client = _create_client(content='{"answer": "ok"}')
    json_schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    provider.generate(
        [{"role": "user", "content": "question"}],
        temperature=0.1,
        max_tokens=128,
        json_schema=json_schema,
    )

    request = provider._client.chat.completions.create.call_args.kwargs
    assert request["model"] == "Qwen/Qwen3-8B"
    assert request["temperature"] == 0.1
    assert request["max_tokens"] == 128
    assert request["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "schema": json_schema,
        },
    }


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (APITimeoutError(request=Mock()), "vLLM не ответил вовремя."),
        (APIConnectionError(request=Mock()), "Не удалось подключиться к vLLM."),
        (
            APIStatusError(
                "Server error",
                response=Mock(status_code=503, request=Mock()),
                body=None,
            ),
            "vLLM вернул ошибку HTTP 503.",
        ),
    ],
)
def test_generate_maps_openai_errors(error: Exception, message: str) -> None:
    provider = VllmProvider(
        "http://localhost:8000/v1",
        "secret",
        "Qwen/Qwen3-8B",
    )
    provider._client = _create_client(error=error)

    with pytest.raises(LlmProviderError, match=message):
        provider.generate([], temperature=0.2, max_tokens=256)


def test_generate_rejects_empty_response() -> None:
    provider = VllmProvider(
        "http://localhost:8000/v1",
        "secret",
        "Qwen/Qwen3-8B",
    )
    provider._client = _create_client(content=None)

    with pytest.raises(LlmProviderError, match="vLLM вернул пустой ответ"):
        provider.generate([], temperature=0.2, max_tokens=256)


def test_generate_requires_api_key_and_model() -> None:
    missing_api_key_provider = VllmProvider("http://localhost:8000/v1", "", "model")
    missing_model_provider = VllmProvider("http://localhost:8000/v1", "secret", "")

    with pytest.raises(LlmProviderError, match="VLLM_API_KEY"):
        missing_api_key_provider.generate([], temperature=0.2, max_tokens=256)

    with pytest.raises(LlmProviderError, match="VLLM_MODEL"):
        missing_model_provider.generate([], temperature=0.2, max_tokens=256)


def _create_client(
    content: str | None = "answer",
    error: Exception | None = None,
) -> Mock:
    client = Mock()

    if error is not None:
        client.chat.completions.create.side_effect = error

        return client

    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=4,
        total_tokens=14,
        model_dump=Mock(return_value={"prompt_tokens": 10, "completion_tokens": 4}),
    )
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=usage,
    )

    return client

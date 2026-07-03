from types import SimpleNamespace

import pytest
from openai.types.chat.chat_completion import ChatCompletion

from tutor_bot.generation.llm_provider_error import LlmProviderError
from tutor_bot.generation.ollama_provider import OllamaProvider
from tutor_bot.generation.yandex_provider import YandexProvider


class _OllamaClient:
    def chat(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            message=SimpleNamespace(content="answer"),
            prompt_eval_count=11,
            eval_count=7,
        )


class _YandexCompletions:
    def __init__(self, response: ChatCompletion) -> None:
        self._response = response

    def create(self, **kwargs: object) -> ChatCompletion:
        return self._response


def test_ollama_provider_returns_normalized_usage() -> None:
    provider = OllamaProvider("http://localhost:11434", "qwen3.5:9b")
    provider._client = _OllamaClient()

    response = provider.generate([], temperature=0.0, max_tokens=100)

    assert response.text == "answer"
    assert response.provider == "ollama"
    assert response.prompt_tokens == 11
    assert response.completion_tokens == 7
    assert response.total_tokens == 18


def test_yandex_provider_returns_normalized_usage() -> None:
    provider = _create_yandex_provider()
    response = ChatCompletion.model_validate(
        {
            "id": "test",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"content": "answer", "role": "assistant"},
                }
            ],
            "created": 0,
            "model": "test",
            "object": "chat.completion",
            "usage": {"prompt_tokens": 13, "completion_tokens": 5, "total_tokens": 18},
        }
    )
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=_YandexCompletions(response))
    )

    result = provider.generate([], temperature=0.0, max_tokens=100)

    assert result.provider == "yandex"
    assert result.model == "qwen3.6-35b-a3b"
    assert result.total_tokens == 18


def test_yandex_provider_rejects_reasoning_without_final_answer() -> None:
    provider = _create_yandex_provider()
    response = ChatCompletion.model_validate(
        {
            "id": "test",
            "choices": [
                {
                    "finish_reason": "length",
                    "index": 0,
                    "message": {
                        "content": None,
                        "role": "assistant",
                        "reasoning_content": "internal reasoning",
                    },
                }
            ],
            "created": 0,
            "model": "test",
            "object": "chat.completion",
        }
    )
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=_YandexCompletions(response))
    )

    with pytest.raises(LlmProviderError, match="YANDEX_MAX_TOKENS"):
        provider.generate([], temperature=0.0, max_tokens=100)


def _create_yandex_provider() -> YandexProvider:
    return YandexProvider(
        "https://ai.api.cloud.yandex.net/v1",
        "secret",
        "folder",
        "qwen3.6-35b-a3b",
        max_tokens=2000,
        temperature=0.3,
    )

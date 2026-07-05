from collections.abc import Callable
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from tutor_bot.generation.llm_provider_error import LlmProviderError
from tutor_bot.generation.llm_response import LlmResponse


class VllmProvider:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        timeout_seconds: float = 120.0,
        usage_callback: Callable[[LlmResponse], None] | None = None,
    ) -> None:
        self._client = OpenAI(
            api_key=api_key or "not-configured",
            base_url=base_url,
            timeout=timeout_seconds,
        )
        self._api_key = api_key
        self._model_name = model_name
        self._usage_callback = usage_callback

    @property
    def provider_name(self) -> str:
        return "vllm"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_schema: dict[str, object] | None = None,
    ) -> LlmResponse:
        if not self._api_key:
            raise LlmProviderError("Не задан VLLM_API_KEY в .env.")

        if not self._model_name:
            raise LlmProviderError("Не задан VLLM_MODEL в .env.")

        request_options: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_schema is not None:
            request_options["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": json_schema,
                },
            }

        try:
            response = self._client.chat.completions.create(**request_options)
        except APITimeoutError as error:
            raise LlmProviderError("vLLM не ответил вовремя.") from error
        except APIConnectionError as error:
            raise LlmProviderError("Не удалось подключиться к vLLM.") from error
        except APIStatusError as error:
            raise LlmProviderError(f"vLLM вернул ошибку HTTP {error.status_code}.") from error

        message = response.choices[0].message

        if not message.content:
            raise LlmProviderError("vLLM вернул пустой ответ.")

        usage = response.usage
        result = LlmResponse(
            text=message.content,
            provider=self.provider_name,
            model=self._model_name,
            prompt_tokens=usage.prompt_tokens if usage is not None else 0,
            completion_tokens=usage.completion_tokens if usage is not None else 0,
            total_tokens=usage.total_tokens if usage is not None else 0,
            raw_usage=usage.model_dump() if usage is not None else None,
        )

        if self._usage_callback is not None:
            self._usage_callback(result)

        return result

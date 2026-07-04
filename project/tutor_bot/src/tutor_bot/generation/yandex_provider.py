from collections.abc import Callable

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from tutor_bot.generation.llm_provider_error import LlmProviderError
from tutor_bot.generation.llm_response import LlmResponse


class YandexProvider:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        folder_id: str,
        model_name: str,
        max_tokens: int,
        temperature: float,
        timeout_seconds: float = 120.0,
        usage_callback: Callable[[LlmResponse], None] | None = None,
    ) -> None:
        self._client = OpenAI(
            api_key=api_key or "not-configured",
            base_url=base_url,
            timeout=timeout_seconds,
        )
        self._model_name = model_name
        self._model_uri = f"gpt://{folder_id}/{model_name}"
        self._api_key = api_key
        self._folder_id = folder_id
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._usage_callback = usage_callback

    @property
    def provider_name(self) -> str:
        return "yandex"

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
            raise LlmProviderError("Не задан YANDEX_API_KEY в .env.")

        if not self._folder_id:
            raise LlmProviderError("Не задан YANDEX_FOLDER_ID в .env.")

        request_options = {
            "model": self._model_uri,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": max(self._max_tokens, max_tokens),
        }

        if json_schema is not None:
            request_options["response_format"] = {"type": "json_object"}

        try:
            response = self._client.chat.completions.create(**request_options)
        except APITimeoutError as error:
            raise LlmProviderError("Yandex AI Studio не ответила вовремя.") from error
        except APIConnectionError as error:
            raise LlmProviderError("Не удалось подключиться к Yandex AI Studio.") from error
        except APIStatusError as error:
            raise self._map_status_error(error) from error

        message = response.choices[0].message

        if not message.content:
            reasoning_content = getattr(message, "reasoning_content", None)

            if reasoning_content is None and message.model_extra is not None:
                reasoning_content = message.model_extra.get("reasoning_content")

            if reasoning_content:
                raise LlmProviderError(
                    "Yandex AI Studio вернула только reasoning без финального ответа. "
                    "Увеличьте YANDEX_MAX_TOKENS в .env и повторите запрос."
                )

            raise LlmProviderError("Yandex AI Studio вернула пустой ответ.")

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

    def _map_status_error(self, error: APIStatusError) -> LlmProviderError:
        if error.status_code in (401, 403):
            return LlmProviderError(
                "Yandex AI Studio отклонила доступ. Проверьте API-ключ, folder ID и права сервисного аккаунта."
            )

        if error.status_code == 400:
            return LlmProviderError(
                "Yandex AI Studio не нашла или не смогла загрузить модель. "
                "Проверьте YANDEX_FOLDER_ID и YANDEX_MODEL."
            )

        return LlmProviderError(f"Yandex AI Studio вернула ошибку HTTP {error.status_code}.")

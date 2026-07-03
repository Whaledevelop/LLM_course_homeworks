from collections.abc import Callable

from ollama import Client, ResponseError

from tutor_bot.generation.llm_provider_error import LlmProviderError
from tutor_bot.generation.llm_response import LlmResponse


class OllamaProvider:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        think: bool = False,
        timeout_seconds: float = 120.0,
        usage_callback: Callable[[LlmResponse], None] | None = None,
    ) -> None:
        self._client = Client(
            host=base_url.removesuffix("/v1").rstrip("/"),
            timeout=timeout_seconds,
        )
        self._model_name = model_name
        self._think = think
        self._usage_callback = usage_callback

    @property
    def provider_name(self) -> str:
        return "ollama"

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
        try:
            response = self._client.chat(
                model=self._model_name,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
                format=json_schema,
                think=self._think,
            )
        except ResponseError as error:
            raise LlmProviderError(f"Ollama отклонила запрос: {error}") from error
        except Exception as error:
            raise LlmProviderError(
                "Ollama недоступна. Проверьте, что сервис запущен и модель загружена."
            ) from error

        text = response.message.content

        if not text:
            raise LlmProviderError("Ollama вернула пустой ответ.")

        result = LlmResponse(
            text=text,
            provider=self.provider_name,
            model=self._model_name,
            prompt_tokens=response.prompt_eval_count or 0,
            completion_tokens=response.eval_count or 0,
            total_tokens=(response.prompt_eval_count or 0) + (response.eval_count or 0),
            raw_usage={
                "prompt_eval_count": response.prompt_eval_count,
                "eval_count": response.eval_count,
            },
        )

        if self._usage_callback is not None:
            self._usage_callback(result)

        return result

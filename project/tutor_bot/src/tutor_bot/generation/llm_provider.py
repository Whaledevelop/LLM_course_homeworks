from typing import Protocol

from tutor_bot.generation.llm_response import LlmResponse


class LlmProvider(Protocol):
    @property
    def provider_name(self) -> str:
        pass

    @property
    def model_name(self) -> str:
        pass

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_schema: dict[str, object] | None = None,
    ) -> LlmResponse:
        pass

from typing import Any

from tutor_bot.application.observability_event_service import (
    ObservabilityEventService,
    add_current_observation_metadata,
    is_generation_observation_active,
)
from tutor_bot.generation.llm_provider import LlmProvider
from tutor_bot.generation.llm_response import LlmResponse


class ObservedLlmProvider:
    def __init__(
        self,
        provider: LlmProvider,
        observability_event_service: ObservabilityEventService,
    ) -> None:
        self._provider = provider
        self._observability_event_service = observability_event_service

    @property
    def provider_name(self) -> str:
        return self._provider.provider_name

    @property
    def model_name(self) -> str:
        return self._provider.model_name

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_schema: dict[str, object] | None = None,
    ) -> LlmResponse:
        generation_payload: dict[str, Any] = {
            "provider": self.provider_name,
            "model": self.model_name,
            "input": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "structured_output": json_schema is not None,
        }

        if is_generation_observation_active():
            add_current_observation_metadata(**generation_payload)
            response = self._generate(
                messages,
                temperature,
                max_tokens,
                json_schema,
            )
            self._add_response_metadata(response)

            return response

        with self._observability_event_service.observe(
            "llm_generation",
            "generation",
            observation_type="generation",
            payload=generation_payload,
        ) as generation_scope:
            response = self._generate(
                messages,
                temperature,
                max_tokens,
                json_schema,
            )
            self._add_response_metadata(response)
            generation_scope.set_output(response.text)

        return response

    def _generate(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_schema: dict[str, object] | None,
    ) -> LlmResponse:
        return self._provider.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=json_schema,
        )

    def _add_response_metadata(self, response: LlmResponse) -> None:
        add_current_observation_metadata(
            provider=response.provider,
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
        )

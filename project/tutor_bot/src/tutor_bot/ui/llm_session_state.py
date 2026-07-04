import streamlit as st
from typing import Literal

from pydantic import BaseModel, ConfigDict

from tutor_bot.config import get_settings
from tutor_bot.application.observability_event_service import (
    add_current_observation_metadata,
)
from tutor_bot.generation.llm_response import LlmResponse
from tutor_bot.infrastructure.atomic_json import write_json_atomically
from tutor_bot.infrastructure.llm_usage_repository import LlmUsageRepository


_PROVIDER_KEY = "llm_provider"
_MODEL_KEY = "llm_model"
_PREFERENCES_FILE_NAME = "llm_preferences.json"


class _LlmPreferences(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_provider: Literal["ollama", "yandex"]
    default_model: str | None = None


def _get_usage_repository() -> LlmUsageRepository:
    return LlmUsageRepository(get_settings().llm_usage_file)


def get_active_provider(default_provider: str = "ollama") -> str:
    if _PROVIDER_KEY not in st.session_state:
        st.session_state[_PROVIDER_KEY] = get_default_provider(default_provider)

    return st.session_state[_PROVIDER_KEY]


def set_active_provider(provider: str) -> None:
    st.session_state[_PROVIDER_KEY] = provider


def get_active_model(default_model: str) -> str:
    if _MODEL_KEY not in st.session_state:
        st.session_state[_MODEL_KEY] = get_default_model(default_model)

    return st.session_state[_MODEL_KEY]


def set_active_model(model: str) -> None:
    st.session_state[_MODEL_KEY] = model


def get_default_provider(fallback_provider: str = "ollama") -> str:
    preferences_file = get_settings().data_dir / _PREFERENCES_FILE_NAME

    if not preferences_file.is_file():
        return fallback_provider

    preferences = _LlmPreferences.model_validate_json(
        preferences_file.read_text(encoding="utf-8-sig")
    )

    return preferences.default_provider


def set_default_provider(provider: str) -> None:
    set_default_llm(provider, None)


def get_default_model(fallback_model: str) -> str:
    preferences_file = get_settings().data_dir / _PREFERENCES_FILE_NAME

    if not preferences_file.is_file():
        return fallback_model

    preferences = _LlmPreferences.model_validate_json(
        preferences_file.read_text(encoding="utf-8-sig")
    )

    return preferences.default_model or fallback_model


def set_default_llm(provider: str, model: str | None) -> None:
    preferences_file = get_settings().data_dir / _PREFERENCES_FILE_NAME
    write_json_atomically(
        preferences_file,
        _LlmPreferences(
            default_provider=provider,
            default_model=model,
        ),
    )


def record_usage(response: LlmResponse) -> None:
    _get_usage_repository().save(response)
    add_current_observation_metadata(
        provider=response.provider,
        model=response.model,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        total_tokens=response.total_tokens,
    )


def get_usage() -> tuple[dict[str, object], ...]:
    return tuple(response.model_dump() for response in _get_usage_repository().load())

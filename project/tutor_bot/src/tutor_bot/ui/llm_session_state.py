import streamlit as st

from tutor_bot.config import get_settings
from tutor_bot.generation.llm_response import LlmResponse
from tutor_bot.infrastructure.llm_usage_repository import LlmUsageRepository


_PROVIDER_KEY = "llm_provider"


def _get_usage_repository() -> LlmUsageRepository:
    return LlmUsageRepository(get_settings().llm_usage_file)


def get_active_provider(default_provider: str = "ollama") -> str:
    if _PROVIDER_KEY not in st.session_state:
        st.session_state[_PROVIDER_KEY] = default_provider

    return st.session_state[_PROVIDER_KEY]


def set_active_provider(provider: str) -> None:
    st.session_state[_PROVIDER_KEY] = provider


def record_usage(response: LlmResponse) -> None:
    _get_usage_repository().save(response)


def get_usage() -> tuple[dict[str, object], ...]:
    return tuple(response.model_dump() for response in _get_usage_repository().load())

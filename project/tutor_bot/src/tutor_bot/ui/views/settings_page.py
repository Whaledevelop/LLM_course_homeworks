import streamlit as st

from tutor_bot.config import get_settings
from tutor_bot.ui.llm_session_state import (
    get_active_provider,
    get_active_model,
    get_default_model,
    get_default_provider,
    set_active_model,
    set_active_provider,
    set_default_llm,
)
from tutor_bot.ui.views.tokens_statistics_page import render_tokens_statistics_page


_LLM_OPTIONS = {
    "ollama": ("ollama", "qwen3.5:9b", "Local Ollama — Qwen3.5 9B"),
    "yandex-qwen-35b": (
        "yandex",
        "qwen3.6-35b-a3b",
        "Yandex AI Studio — Qwen3.6 35B",
    ),
    "yandex-gpt-oss-120b": (
        "yandex",
        "gpt-oss-120b",
        "Yandex AI Studio — GPT-OSS 120B",
    ),
    "yandex-qwen-235b": (
        "yandex",
        "qwen3-235b-a22b-fp8",
        "Yandex AI Studio — Qwen3 235B",
    ),
    "yandex-deepseek-v4": (
        "yandex",
        "deepseek-v4-flash",
        "Yandex AI Studio — DeepSeek V4 Flash",
    ),
}


def render_llms_page() -> None:
    settings = get_settings()
    active_provider = get_active_provider(settings.llm_provider)
    active_model = get_active_model(
        settings.yandex_model if active_provider == "yandex" else settings.ollama_model
    )
    active_option = next(
        (
            option
            for option, (provider, model, _) in _LLM_OPTIONS.items()
            if provider == active_provider and model == active_model
        ),
        "ollama",
    )
    selected_option = st.selectbox(
        "LLM",
        options=tuple(_LLM_OPTIONS),
        index=tuple(_LLM_OPTIONS).index(active_option),
        format_func=lambda option: _LLM_OPTIONS[option][2],
    )
    selected_provider, selected_model, _ = _LLM_OPTIONS[selected_option]
    set_active_provider(selected_provider)
    set_active_model(selected_model)

    default_provider = get_default_provider(settings.llm_provider)
    default_model = get_default_model(
        settings.yandex_model if default_provider == "yandex" else settings.ollama_model
    )

    if selected_provider == default_provider and selected_model == default_model:
        st.caption("Используется по умолчанию")
    elif st.button("Использовать по умолчанию"):
        set_default_llm(selected_provider, selected_model)
        st.success("LLM по умолчанию обновлена")

    if selected_provider == "yandex" and (
        not settings.yandex_api_key or not settings.yandex_folder_id
    ):
        st.error("Для Yandex AI Studio заполните YANDEX_API_KEY и YANDEX_FOLDER_ID в .env.")

    st.divider()
    st.subheader("Tokens Statistics")
    render_tokens_statistics_page()

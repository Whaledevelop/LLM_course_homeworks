import streamlit as st

from tutor_bot.config import Settings, get_settings
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


def render_llms_page() -> None:
    settings = get_settings()
    llm_options = _get_llm_options(settings)
    active_provider = get_active_provider(settings.llm_provider)
    active_model = get_active_model(_get_provider_model(settings, active_provider))
    active_option = next(
        (
            option
            for option, (provider, model, _) in llm_options.items()
            if provider == active_provider and model == active_model
        ),
        "ollama",
    )
    selected_option = st.selectbox(
        "LLM",
        options=tuple(llm_options),
        index=tuple(llm_options).index(active_option),
        format_func=lambda option: llm_options[option][2],
    )
    selected_provider, selected_model, _ = llm_options[selected_option]
    set_active_provider(selected_provider)
    set_active_model(selected_model)

    default_provider = get_default_provider(settings.llm_provider)
    default_model = get_default_model(_get_provider_model(settings, default_provider))

    if selected_provider == default_provider and selected_model == default_model:
        st.caption("Используется по умолчанию")
    elif st.button("Использовать по умолчанию"):
        set_default_llm(selected_provider, selected_model)
        st.success("LLM по умолчанию обновлена")

    if selected_provider == "yandex" and (
        not settings.yandex_api_key or not settings.yandex_folder_id
    ):
        st.error("Для Yandex AI Studio заполните YANDEX_API_KEY и YANDEX_FOLDER_ID в .env.")

    if selected_provider == "vllm" and (not settings.vllm_api_key or not settings.vllm_model):
        st.error("Для vLLM заполните VLLM_API_KEY и VLLM_MODEL в .env.")

    st.divider()
    st.subheader("Tokens Statistics")
    render_tokens_statistics_page()


def _get_llm_options(settings: Settings) -> dict[str, tuple[str, str, str]]:
    vllm_label = f"vLLM — {settings.vllm_model or 'модель не настроена'}"

    return {
        "ollama": ("ollama", "qwen3.5:9b", "Local Ollama — Qwen3.5 9B"),
        "vllm": ("vllm", settings.vllm_model, vllm_label),
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


def _get_provider_model(settings: Settings, provider_name: str) -> str:
    if provider_name == "yandex":
        return settings.yandex_model

    if provider_name == "vllm":
        return settings.vllm_model

    return settings.ollama_model

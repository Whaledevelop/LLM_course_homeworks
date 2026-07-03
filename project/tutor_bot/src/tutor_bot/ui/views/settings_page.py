import streamlit as st

from tutor_bot.config import get_settings
from tutor_bot.ui.llm_session_state import get_active_provider, set_active_provider


_PROVIDER_LABELS = {
    "ollama": "Local Ollama — Qwen3.5:9B",
    "yandex": "Yandex AI Studio — qwen3.6:35b",
}


def render_settings_page() -> None:
    settings = get_settings()
    st.title("Settings")
    st.subheader("LLM")
    active_provider = get_active_provider(settings.llm_provider)
    selected_provider = st.radio(
        "Active LLM",
        options=tuple(_PROVIDER_LABELS),
        index=tuple(_PROVIDER_LABELS).index(active_provider),
        format_func=lambda provider: _PROVIDER_LABELS[provider],
    )
    set_active_provider(selected_provider)

    if selected_provider == "yandex" and (
        not settings.yandex_api_key or not settings.yandex_folder_id
    ):
        st.error("Для Yandex AI Studio заполните YANDEX_API_KEY и YANDEX_FOLDER_ID в .env.")

    st.caption("Выбор действует в текущей сессии. По умолчанию используется локальная Ollama.")

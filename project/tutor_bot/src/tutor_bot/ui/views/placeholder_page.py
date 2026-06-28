import streamlit as st

from tutor_bot.ui.app_mode import AppMode


def render_placeholder_page(app_mode: AppMode) -> None:
    st.header(app_mode.value)
    st.info("Интерфейс этого режима будет добавлен на следующих этапах.")

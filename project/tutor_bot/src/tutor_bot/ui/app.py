import streamlit as st

from tutor_bot.application.note_query_service import NoteQueryService
from tutor_bot.infrastructure.mock_note_query_service import MockNoteQueryService
from tutor_bot.ui.app_mode import AppMode
from tutor_bot.ui.views.browse_notes_page import render_browse_notes_page
from tutor_bot.ui.views.placeholder_page import render_placeholder_page


def main() -> None:
    st.set_page_config(
        page_title="Tutor Bot",
        page_icon="🎓",
        layout="wide",
    )

    st.title("Tutor Bot")
    st.caption("Локальный персонализированный учебный ассистент")

    selected_mode = st.sidebar.radio(
        "Режим работы",
        options=list(AppMode),
        format_func=lambda app_mode: app_mode.value,
    )

    note_query_service: NoteQueryService = MockNoteQueryService()

    if selected_mode == AppMode.BROWSE_NOTES:
        render_browse_notes_page(note_query_service)

        return

    render_placeholder_page(selected_mode)


if __name__ == "__main__":
    main()

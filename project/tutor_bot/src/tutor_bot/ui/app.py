import streamlit as st

from tutor_bot.application.note_command_service import NoteCommandService
from tutor_bot.application.note_query_service import NoteQueryService
from tutor_bot.config import get_settings
from tutor_bot.infrastructure.file_note_command_service import (
    FileNoteCommandService,
)
from tutor_bot.infrastructure.metadata_note_query_service import (
    MetadataNoteQueryService,
)
from tutor_bot.infrastructure.notes_metadata_repository import (
    NotesMetadataRepository,
)
from tutor_bot.ui.app_mode import AppMode
from tutor_bot.ui.tutor_answer_service_factory import (
    create_active_recall_service,
    create_assignment_review_service,
    create_tutor_answer_service,
)
from tutor_bot.ui.views.active_recall_page import render_active_recall_page
from tutor_bot.ui.views.add_note_page import render_add_note_page
from tutor_bot.ui.views.assignment_review_page import render_assignment_review_page
from tutor_bot.ui.views.browse_notes_page import render_browse_notes_page
from tutor_bot.ui.views.placeholder_page import render_placeholder_page
from tutor_bot.ui.views.questions_page import render_questions_page


@st.cache_resource
def create_note_services() -> tuple[
    NoteQueryService,
    NoteCommandService,
]:
    settings = get_settings()
    metadata_repository = NotesMetadataRepository(settings.metadata_file)

    query_service = MetadataNoteQueryService(
        metadata_repository,
        settings.source_notes_dir,
    )

    command_service = FileNoteCommandService(
        metadata_repository,
        settings.source_notes_dir,
    )

    return query_service, command_service


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

    if selected_mode == AppMode.QUESTIONS:
        render_questions_page(create_tutor_answer_service())

        return

    if selected_mode == AppMode.ASSIGNMENT_REVIEW:
        render_assignment_review_page(create_assignment_review_service())

        return

    if selected_mode == AppMode.ACTIVE_RECALL:
        note_query_service, _ = create_note_services()
        render_active_recall_page(
            note_query_service,
            create_active_recall_service(note_query_service),
        )

        return

    note_query_service, note_command_service = create_note_services()

    if selected_mode == AppMode.ADD_NOTE:
        render_add_note_page(note_command_service)

        return

    if selected_mode == AppMode.BROWSE_NOTES:
        render_browse_notes_page(
            note_query_service,
            note_command_service,
        )

        return

    render_placeholder_page(selected_mode)


if __name__ == "__main__":
    main()

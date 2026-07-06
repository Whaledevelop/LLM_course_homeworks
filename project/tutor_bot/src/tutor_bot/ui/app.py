import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tutor_bot.application.note_command_service import NoteCommandService
from tutor_bot.application.note_query_service import NoteQueryService
from tutor_bot.config import get_settings
from tutor_bot.infrastructure.file_note_command_service import (
    FileNoteCommandService,
)
from tutor_bot.infrastructure.active_database_service import ActiveDatabaseService
from tutor_bot.infrastructure.database_notes_repository import DatabaseNotesRepository
from tutor_bot.infrastructure.metadata_note_query_service import (
    MetadataNoteQueryService,
)
from tutor_bot.infrastructure.ui_state_repository import UiStateRepository
from tutor_bot.ui.app_mode import APP_MODE_STATE_KEY, AppMode, apply_pending_app_mode
from tutor_bot.ui.tutor_answer_service_factory import (
    create_active_recall_service,
    create_assignment_review_service,
    create_chat_service,
    create_note_metadata_suggester,
    create_note_content_generator,
    create_vacancy_analyzer,
    create_vacancy_matching_service,
    create_vacancy_preparation_service,
)
from tutor_bot.ui.views.active_recall_page import (
    interrupt_active_recall_session,
    render_active_recall_page,
)
from tutor_bot.ui.views.add_note_page import render_add_note_page
from tutor_bot.ui.views.assignment_review_page import render_assignment_review_page
from tutor_bot.ui.views.browse_notes_page import (
    render_browse_notes_page,
    request_browse_page_scroll_to_top,
)
from tutor_bot.ui.views.databases_page import render_databases_page
from tutor_bot.ui.views.placeholder_page import render_placeholder_page
from tutor_bot.ui.views.questions_page import render_questions_page
from tutor_bot.ui.views.settings_page import render_llms_page
from tutor_bot.ui.views.prepare_for_vacancy_page import (
    interrupt_vacancy_session,
    render_prepare_for_vacancy_page,
)
from tutor_bot.infrastructure.vacancy_repository import VacancyRepository


_VISIBLE_APP_MODES = [
    AppMode.BROWSE_NOTES,
    AppMode.ADD_NOTE,
    AppMode.DATABASES,
    AppMode.TEST_NOTES,
    AppMode.LLMS,
    AppMode.PREPARE_FOR_VACANCY,
    AppMode.QUESTIONS,
]


@st.cache_resource
def create_note_services(
    db_id: str, root_path: str
) -> tuple[
    NoteQueryService,
    NoteCommandService,
]:
    settings = get_settings()
    metadata_repository = DatabaseNotesRepository(
        settings.data_dir / "metadata",
        db_id,
        Path(root_path),
    )

    query_service = MetadataNoteQueryService(
        metadata_repository,
        Path(root_path),
    )

    command_service = FileNoteCommandService(
        metadata_repository,
        Path(root_path),
    )

    return query_service, command_service


@st.cache_resource
def create_ui_state_repository(db_id: str) -> UiStateRepository:
    settings = get_settings()

    return UiStateRepository(settings.data_dir / "ui_state" / f"{db_id}.json")


def main() -> None:
    st.set_page_config(
        page_title="Tutor Bot",
        page_icon="🎓",
        layout="wide",
    )
    apply_pending_app_mode()
    _apply_styles()

    st.sidebar.title("Tutor Bot")

    selected_mode = st.sidebar.radio(
        "Режим работы",
        options=_VISIBLE_APP_MODES,
        format_func=lambda app_mode: app_mode.value,
        key=APP_MODE_STATE_KEY,
        on_change=_interrupt_study_sessions,
        label_visibility="collapsed",
    )

    active_database = ActiveDatabaseService(get_settings().data_dir).get_active()

    if selected_mode == AppMode.LLMS:
        render_llms_page()

        return

    if active_database is None and selected_mode not in {
        AppMode.LLMS,
        AppMode.DATABASES,
    }:
        st.error("Активная DB не выбрана. Создайте или выберите DB в режиме «Базы данных».")

        return

    if selected_mode == AppMode.QUESTIONS:
        note_query_service, note_command_service = create_note_services(
            active_database.db_id,
            str(active_database.root_path),
        )
        render_questions_page(
            lambda: create_chat_service(
                active_database.db_id,
                note_query_service,
                note_command_service,
            ),
        )

        return

    if selected_mode == AppMode.ASSIGNMENT_REVIEW:
        render_assignment_review_page(create_assignment_review_service(active_database.db_id))

        return

    if selected_mode == AppMode.TEST_NOTES:
        note_query_service, note_command_service = create_note_services(
            active_database.db_id,
            str(active_database.root_path),
        )
        render_active_recall_page(
            note_query_service,
            create_active_recall_service(
                note_query_service,
                note_command_service,
                active_database.db_id,
            ),
        )

        return

    if selected_mode == AppMode.PREPARE_FOR_VACANCY:
        note_query_service, note_command_service = create_note_services(
            active_database.db_id,
            str(active_database.root_path),
        )
        render_prepare_for_vacancy_page(
            VacancyRepository(
                get_settings().data_dir / "vacancies",
                active_database.db_id,
            ),
            create_vacancy_analyzer,
            lambda: create_vacancy_matching_service(active_database.db_id),
            lambda: create_vacancy_preparation_service(
                note_query_service,
                note_command_service,
                active_database.db_id,
            ),
        )

        return

    if selected_mode == AppMode.DATABASES:
        render_databases_page(ActiveDatabaseService(get_settings().data_dir))

        return

    note_query_service, note_command_service = create_note_services(
        active_database.db_id,
        str(active_database.root_path),
    )

    if selected_mode == AppMode.ADD_NOTE:
        render_add_note_page(
            note_command_service,
            note_query_service,
            create_note_metadata_suggester(),
            create_note_content_generator(),
        )

        return

    if selected_mode == AppMode.BROWSE_NOTES:
        render_browse_notes_page(
            note_query_service,
            note_command_service,
            create_ui_state_repository(active_database.db_id),
            create_active_recall_service(
                note_query_service,
                note_command_service,
                active_database.db_id,
            ),
            create_note_content_generator(),
        )

        return

    render_placeholder_page(selected_mode)


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stButton"] > button,
        div[data-testid="stFormSubmitButton"] > button {
            background-color: #285943;
            border-color: #357257;
            color: #ffffff;
        }
        div[data-testid="stButton"] > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            background-color: #1f4735;
            border-color: #4b8a6b;
            color: #ffffff;
        }
        div[data-testid="stButton"] > button:focus-visible,
        div[data-testid="stFormSubmitButton"] > button:focus-visible {
            box-shadow: 0 0 0 0.2rem rgba(75, 138, 107, 0.35);
        }
        div[data-testid="stButton"] > button[kind="tertiary"] {
            background-color: #4b4f4d;
            border-color: #606562;
            color: #ffffff;
        }
        div[data-testid="stButton"] > button[kind="tertiary"]:hover {
            background-color: #3e4240;
            border-color: #737a76;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _interrupt_study_sessions() -> None:
    interrupt_active_recall_session()
    interrupt_vacancy_session()

    if st.session_state.get(APP_MODE_STATE_KEY) == AppMode.BROWSE_NOTES:
        request_browse_page_scroll_to_top()


if __name__ == "__main__":
    main()

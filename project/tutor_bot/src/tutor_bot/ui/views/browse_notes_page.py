from uuid import UUID

import streamlit as st

from tutor_bot.application.note_command_service import NoteCommandService
from tutor_bot.application.active_recall_service import ActiveRecallService
from tutor_bot.application.note_list_item import NoteListItem
from tutor_bot.application.note_query_service import NoteQueryService
from tutor_bot.infrastructure.ui_state_repository import UiStateRepository
from tutor_bot.ui.views.note_actions import render_note_actions
from tutor_bot.ui.views.active_recall_page import start_note_test
from tutor_bot.generation.note_content_generator import NoteContentGenerator


_SUCCESS_MESSAGE_KEY = "note_action_success"
_TABLE_VERSION_KEY = "notes_table_version"
_SELECTED_NOTE_ID_KEY = "selected_browse_note_id"


def render_browse_notes_page(
    note_query_service: NoteQueryService,
    note_command_service: NoteCommandService,
    ui_state_repository: UiStateRepository,
    recall_service: ActiveRecallService,
    content_generator: NoteContentGenerator,
) -> None:
    success_message = st.session_state.pop(
        _SUCCESS_MESSAGE_KEY,
        "",
    )

    if success_message:
        st.success(success_message)

    notes = note_query_service.list_notes()

    if not notes:
        st.info("Заметки пока отсутствуют.")

        return

    content_column, list_column = st.columns(
        [0.68, 0.32],
        gap="large",
    )

    groups = sorted({note.group for note in notes if note.group})

    with list_column:
        st.subheader("Заметки")
        search_text = st.text_input(
            "Поиск",
            placeholder="Название или тема",
        )
        selected_group = st.selectbox(
            "Группа",
            options=["Все", *groups],
        )

        filtered_notes = _filter_notes(
            notes,
            search_text,
            selected_group,
        )

        if not filtered_notes:
            st.warning("По заданным условиям заметки не найдены.")

            return

        st.caption(f"Показано: {len(filtered_notes)} из {len(notes)}")
        selected_note = _render_notes_table(filtered_notes)

    if selected_note is None:
        selected_note = _resolve_selected_note(
            notes,
            ui_state_repository,
        )

    if selected_note is None:
        return

    st.session_state[_SELECTED_NOTE_ID_KEY] = str(selected_note.id)
    ui_state_repository.save_selected_note_id(selected_note.id)

    note_details = note_query_service.get_note(selected_note.id)

    with content_column:
        st.subheader(note_details.title)
        action_message = render_note_actions(
            note_details,
            note_command_service,
            lambda note_id: start_note_test(recall_service, note_id),
            content_generator,
        )

        st.caption(
            f"Группа: {note_details.group or 'не указана'} · "
            f"Важность: {note_details.importance} · "
            f"Знание: {note_details.knowledge} · "
            f"Заполненность: {note_details.fullness}"
        )

        if action_message:
            st.session_state[_SUCCESS_MESSAGE_KEY] = action_message
            st.session_state[_TABLE_VERSION_KEY] = st.session_state.get(_TABLE_VERSION_KEY, 0) + 1
            st.rerun()

        if note_details.comment:
            st.info(note_details.comment)

        st.markdown(note_details.markdown_content)


def _filter_notes(
    notes: list[NoteListItem],
    search_text: str,
    selected_group: str,
) -> list[NoteListItem]:
    normalized_search = search_text.strip().casefold()
    filtered_notes = []

    for note in notes:
        searchable_text = " ".join(
            [
                note.title,
                note.group,
            ]
        ).casefold()

        matches_search = not normalized_search or normalized_search in searchable_text
        matches_group = selected_group == "Все" or note.group == selected_group

        if matches_search and matches_group:
            filtered_notes.append(note)

    return filtered_notes


def _render_notes_table(
    filtered_notes: list[NoteListItem],
) -> NoteListItem | None:
    rows = [
        {
            "Название": f"🔗 {note.title}",
            "Группа": note.group or "не указана",
        }
        for note in filtered_notes
    ]

    table_version = st.session_state.get(_TABLE_VERSION_KEY, 0)

    table_state = st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        height=720,
        key=f"notes_table_{table_version}",
        on_select="rerun",
        selection_mode="single-cell",
    )

    selected_cells = table_state.selection.cells

    if not selected_cells:
        return None

    selected_note_index, selected_column = selected_cells[0]

    if selected_column != "Название":
        return None

    return filtered_notes[selected_note_index]


def _resolve_selected_note(
    notes: list[NoteListItem],
    ui_state_repository: UiStateRepository,
) -> NoteListItem | None:
    note_by_id = {note.id: note for note in notes}
    cached_note_id = st.session_state.get(_SELECTED_NOTE_ID_KEY)

    if cached_note_id:
        selected_note = note_by_id.get(UUID(str(cached_note_id)))

        if selected_note is not None:
            return selected_note

    persisted_note_id = ui_state_repository.load_selected_note_id()

    if persisted_note_id is not None:
        selected_note = note_by_id.get(persisted_note_id)

        if selected_note is not None:
            return selected_note

    return notes[0] if notes else None

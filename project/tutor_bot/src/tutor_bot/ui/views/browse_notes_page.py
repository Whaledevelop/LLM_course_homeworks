from uuid import UUID

import streamlit as st

from tutor_bot.application.note_command_service import NoteCommandService
from tutor_bot.application.note_list_item import NoteListItem
from tutor_bot.application.note_query_service import NoteQueryService
from tutor_bot.infrastructure.ui_state_repository import UiStateRepository
from tutor_bot.ui.views.note_actions import render_note_actions


_SUCCESS_MESSAGE_KEY = "note_action_success"
_TABLE_VERSION_KEY = "notes_table_version"
_SELECTED_NOTE_ID_KEY = "selected_browse_note_id"


def render_browse_notes_page(
    note_query_service: NoteQueryService,
    note_command_service: NoteCommandService,
    ui_state_repository: UiStateRepository,
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

    search_text = st.text_input(
        "Поиск",
        placeholder="Название или тема",
    )

    content_column, list_column = st.columns(
        [0.68, 0.32],
        gap="large",
    )

    themes = sorted({note.theme for note in notes if note.theme})

    with list_column:
        with st.expander(
            "Заметки",
            expanded=True,
        ):
            selected_theme = st.selectbox(
                "Тема",
                options=["Все", *themes],
            )

            filtered_notes = _filter_notes(
                notes,
                search_text,
                selected_theme,
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
        title_column, actions_column = st.columns([0.86, 0.14])
        title_column.subheader(note_details.title)

        with actions_column:
            action_message = render_note_actions(
                note_details,
                note_command_service,
            )

        st.caption(
            f"Тема: {note_details.theme or 'не указана'} · "
            f"Важность: {note_details.importance} · "
            f"Знание: {note_details.knowledge}"
        )

        if action_message:
            st.session_state[_SUCCESS_MESSAGE_KEY] = action_message
            st.session_state[_TABLE_VERSION_KEY] = (
                st.session_state.get(_TABLE_VERSION_KEY, 0) + 1
            )
            st.rerun()

        if note_details.comment:
            st.info(note_details.comment)

        st.markdown(note_details.markdown_content)


def _filter_notes(
    notes: list[NoteListItem],
    search_text: str,
    selected_theme: str,
) -> list[NoteListItem]:
    normalized_search = search_text.strip().casefold()
    filtered_notes = []

    for note in notes:
        searchable_text = " ".join(
            [
                note.title,
                note.theme,
            ]
        ).casefold()

        matches_search = not normalized_search or normalized_search in searchable_text
        matches_theme = selected_theme == "Все" or note.theme == selected_theme

        if matches_search and matches_theme:
            filtered_notes.append(note)

    return filtered_notes


def _render_notes_table(
    filtered_notes: list[NoteListItem],
) -> NoteListItem | None:
    rows = [
        {
            "Название": f"🔗 {note.title}",
            "Тема": note.theme or "не указана",
        }
        for note in filtered_notes
    ]

    table_version = st.session_state.get(_TABLE_VERSION_KEY, 0)

    table_state = st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
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

import streamlit as st

from tutor_bot.application.note_command_service import NoteCommandService
from tutor_bot.application.note_query_service import NoteQueryService
from tutor_bot.ui.views.note_actions import render_note_actions


_SUCCESS_MESSAGE_KEY = "note_action_success"
_TABLE_VERSION_KEY = "notes_table_version"


def render_browse_notes_page(
    note_query_service: NoteQueryService,
    note_command_service: NoteCommandService,
) -> None:
    st.header("Просмотр и редактирование")

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
        placeholder="Название, тема или сложность",
    )

    themes = sorted({note.theme for note in notes if note.theme})

    selected_theme = st.selectbox(
        "Тема",
        options=["Все", *themes],
    )

    normalized_search = search_text.strip().casefold()
    filtered_notes = []

    for note in notes:
        searchable_text = " ".join(
            [
                note.title,
                note.theme,
                note.difficulty,
            ]
        ).casefold()

        matches_search = not normalized_search or normalized_search in searchable_text
        matches_theme = selected_theme == "Все" or note.theme == selected_theme

        if not (matches_search and matches_theme):
            continue

        filtered_notes.append(note)

    if not filtered_notes:
        st.warning("По заданным условиям заметки не найдены.")

        return

    st.caption(f"Показано заметок: {len(filtered_notes)} из {len(notes)}")

    rows = [
        {
            "Название": f"🔗 {note.title}",
            "Тема": note.theme or "не указана",
            "Сложность": note.difficulty or "не указана",
            "Важность": note.importance,
            "Полнота": note.completeness,
            "Освоение": note.mastery,
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
        st.info("Нажмите на название, чтобы открыть заметку.")

        return

    selected_note_index, selected_column = selected_cells[0]

    if selected_column != "Название":
        st.info("Для открытия заметки нажмите на её название.")

        return

    selected_note = filtered_notes[selected_note_index]
    note_details = note_query_service.get_note(selected_note.id)

    st.divider()
    st.subheader(note_details.title)
    st.caption(
        f"Тема: {note_details.theme or 'не указана'} · "
        f"Сложность: {note_details.difficulty or 'не указана'}"
    )

    action_message = render_note_actions(
        note_details,
        note_command_service,
    )

    if action_message:
        st.session_state[_SUCCESS_MESSAGE_KEY] = action_message
        st.session_state[_TABLE_VERSION_KEY] = table_version + 1
        st.rerun()

    if note_details.comment:
        st.info(note_details.comment)

    st.markdown(note_details.markdown_content)

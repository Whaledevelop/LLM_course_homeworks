import streamlit as st

from tutor_bot.application.note_command_service import NoteCommandService
from tutor_bot.application.note_query_service import NoteQueryService
from tutor_bot.application.update_note_command import UpdateNoteCommand


_SUCCESS_MESSAGE_KEY = "note_update_success"
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
            "Название": note.title,
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
        selection_mode="single-row",
    )

    selected_rows = table_state.selection.rows

    if not selected_rows:
        st.info("Выберите строку таблицы, чтобы открыть заметку.")

        return

    selected_note = filtered_notes[selected_rows[0]]
    note_details = note_query_service.get_note(selected_note.id)

    st.divider()
    st.subheader(note_details.title)
    st.caption(
        f"Тема: {note_details.theme or 'не указана'} · "
        f"Сложность: {note_details.difficulty or 'не указана'}"
    )

    if note_details.comment:
        st.info(note_details.comment)

    st.markdown(note_details.markdown_content)

    with st.expander("Редактировать заметку"):
        with st.form(f"edit-note-{note_details.id}"):
            title = st.text_input(
                "Название",
                value=note_details.title,
            )

            theme = st.text_input(
                "Тема",
                value=note_details.theme,
            )

            difficulty = st.text_input(
                "Сложность",
                value=note_details.difficulty,
            )

            comment = st.text_input(
                "Комментарий",
                value=note_details.comment,
            )

            importance = st.slider(
                "Важность",
                min_value=0,
                max_value=10,
                value=note_details.importance,
            )

            completeness = st.slider(
                "Полнота",
                min_value=0,
                max_value=10,
                value=note_details.completeness,
            )

            mastery = st.slider(
                "Освоение",
                min_value=0,
                max_value=10,
                value=note_details.mastery,
            )

            markdown_content = st.text_area(
                "Markdown",
                value=note_details.markdown_content,
                height=500,
            )

            submitted = st.form_submit_button(
                "Сохранить",
                type="primary",
            )

    if not submitted:
        return

    command = UpdateNoteCommand(
        note_id=note_details.id,
        title=title,
        theme=theme,
        comment=comment,
        difficulty=difficulty,
        importance=importance,
        completeness=completeness,
        mastery=mastery,
        markdown_content=markdown_content,
    )

    updated_note = note_command_service.update_note(command)

    st.session_state[_SUCCESS_MESSAGE_KEY] = f"Заметка «{updated_note.title}» сохранена."

    st.session_state[_TABLE_VERSION_KEY] = table_version + 1
    st.rerun()

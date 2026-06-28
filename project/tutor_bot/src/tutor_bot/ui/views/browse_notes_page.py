import streamlit as st

from tutor_bot.application.note_query_service import NoteQueryService


def render_browse_notes_page(
    note_query_service: NoteQueryService,
) -> None:
    st.header("Просмотр и редактирование")

    notes = note_query_service.list_notes()

    if not notes:
        st.info("Заметки пока отсутствуют.")

        return

    st.caption(f"Найдено заметок: {len(notes)}")

    for note in notes:
        with st.container(border=True):
            st.subheader(note.title)

            theme_column, difficulty_column = st.columns(2)
            theme_column.write(f"Тема: `{note.theme or 'не указана'}`")
            difficulty_column.write(f"Сложность: `{note.difficulty or 'не указана'}`")

            importance_column, completeness_column, mastery_column = st.columns(3)
            importance_column.metric("Важность", note.importance)
            completeness_column.metric("Полнота", note.completeness)
            mastery_column.metric("Освоение", note.mastery)

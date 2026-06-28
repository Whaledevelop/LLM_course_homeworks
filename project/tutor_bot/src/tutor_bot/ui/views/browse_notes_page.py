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

    st.dataframe(
        rows,
        hide_index=True,
        use_container_width=True,
    )

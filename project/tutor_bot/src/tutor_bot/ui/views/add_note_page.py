import streamlit as st

from tutor_bot.application.create_note_command import CreateNoteCommand
from tutor_bot.application.note_command_service import NoteCommandService


def render_add_note_page(
    note_command_service: NoteCommandService,
) -> None:
    st.header("Пополнение базы знаний")
    st.caption("Новая заметка получит стабильный UUID и будет сохранена в локальной базе Markdown.")

    with st.form(
        "create-note",
        clear_on_submit=True,
    ):
        title = st.text_input("Название")
        theme = st.text_input("Тема")
        difficulty = st.text_input(
            "Сложность",
            value="middle",
        )
        comment = st.text_input("Комментарий")

        importance = st.slider(
            "Важность",
            min_value=0,
            max_value=10,
            value=5,
        )

        completeness = st.slider(
            "Полнота",
            min_value=0,
            max_value=10,
            value=0,
        )

        mastery = st.slider(
            "Освоение",
            min_value=0,
            max_value=10,
            value=0,
        )

        markdown_content = st.text_area(
            "Markdown",
            height=500,
            placeholder="# Заголовок\n\nСодержание заметки",
        )

        submitted = st.form_submit_button(
            "Создать заметку",
            type="primary",
        )

    if not submitted:
        return

    if not title.strip() or not markdown_content.strip():
        st.error("Название и содержимое заметки обязательны.")

        return

    command = CreateNoteCommand(
        title=title.strip(),
        theme=theme.strip(),
        comment=comment.strip(),
        difficulty=difficulty.strip(),
        importance=importance,
        completeness=completeness,
        mastery=mastery,
        markdown_content=markdown_content,
    )

    created_note = note_command_service.create_note(command)

    st.success(f"Заметка «{created_note.title}» создана. ID: {created_note.id}")

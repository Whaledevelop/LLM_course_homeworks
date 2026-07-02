import streamlit as st
from httpx import HTTPError
from ollama import ResponseError
from pydantic import ValidationError

from tutor_bot.application.create_note_command import CreateNoteCommand
from tutor_bot.application.note_command_service import NoteCommandService
from tutor_bot.application.note_metadata_suggestion import NoteMetadataSuggestion
from tutor_bot.generation.note_metadata_suggester import NoteMetadataSuggester


_TITLE_KEY = "add_note_title"
_THEME_KEY = "add_note_theme"
_DIFFICULTY_KEY = "add_note_difficulty"
_COMMENT_KEY = "add_note_comment"
_SUGGESTION_KEY = "add_note_metadata_suggestion"
_SUGGESTION_SOURCE_KEY = "add_note_metadata_suggestion_source"


def render_add_note_page(
    note_command_service: NoteCommandService,
    metadata_suggester: NoteMetadataSuggester,
) -> None:
    st.header("Пополнение базы знаний")
    st.caption("Новая заметка получит стабильный UUID и будет сохранена в локальной базе Markdown.")
    st.session_state.setdefault(_DIFFICULTY_KEY, "middle")

    with st.form("metadata-suggestion"):
        markdown_content = st.text_area(
            "Markdown",
            height=500,
            placeholder="# Заголовок\n\nСодержание заметки",
        )
        suggestion_submitted = st.form_submit_button("Предложить метаданные")

    if suggestion_submitted:
        _suggest_metadata(
            metadata_suggester,
            markdown_content,
        )

    suggestion = _get_cached_suggestion(markdown_content.strip())

    if suggestion is not None:
        st.caption(f"Ключевые понятия: {', '.join(suggestion.key_concepts)}")

    with st.form(
        "create-note",
        clear_on_submit=True,
    ):
        title = st.text_input(
            "Название",
            key=_TITLE_KEY,
        )
        theme = st.text_input(
            "Тема",
            key=_THEME_KEY,
        )
        difficulty = st.text_input(
            "Сложность",
            key=_DIFFICULTY_KEY,
        )
        comment = st.text_input(
            "Комментарий",
            key=_COMMENT_KEY,
        )
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


def _suggest_metadata(
    metadata_suggester: NoteMetadataSuggester,
    markdown_content: str,
) -> None:
    normalized_markdown = markdown_content.strip()

    if not normalized_markdown:
        st.error("Сначала добавьте содержимое заметки.")

        return

    suggestion = _get_cached_suggestion(normalized_markdown)

    if suggestion is None:
        try:
            with st.spinner("Анализирую заметку и предлагаю метаданные..."):
                suggestion = metadata_suggester.suggest(normalized_markdown)
        except (HTTPError, ResponseError, RuntimeError, ValidationError) as error:
            st.error(f"Не удалось предложить метаданные: {error}")

            return

        st.session_state[_SUGGESTION_KEY] = suggestion
        st.session_state[_SUGGESTION_SOURCE_KEY] = normalized_markdown

    st.session_state[_TITLE_KEY] = suggestion.title
    st.session_state[_THEME_KEY] = suggestion.theme
    st.session_state[_DIFFICULTY_KEY] = suggestion.difficulty
    st.session_state[_COMMENT_KEY] = suggestion.comment


def _get_cached_suggestion(
    markdown_content: str,
) -> NoteMetadataSuggestion | None:
    if st.session_state.get(_SUGGESTION_SOURCE_KEY) != markdown_content:
        return None

    return st.session_state.get(_SUGGESTION_KEY)

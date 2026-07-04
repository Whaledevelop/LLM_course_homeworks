import streamlit as st
from httpx import HTTPError
from pydantic import ValidationError

from tutor_bot.application.create_note_command import CreateNoteCommand
from tutor_bot.application.note_fullness import estimate_note_fullness
from tutor_bot.application.note_command_service import NoteCommandService
from tutor_bot.application.note_metadata_suggestion import NoteMetadataSuggestion
from tutor_bot.generation.note_metadata_suggester import NoteMetadataSuggester
from tutor_bot.generation.note_content_generator import NoteContentGenerator


_TITLE_KEY = "add_note_title"
_MARKDOWN_KEY = "add_note_markdown"
_GROUP_KEY = "add_note_group"
_COMMENT_KEY = "add_note_comment"
_SUGGESTION_KEY = "add_note_metadata_suggestion"
_SUGGESTION_SOURCE_KEY = "add_note_metadata_suggestion_source"
_PENDING_SUGGESTION_KEY = "add_note_pending_metadata_suggestion"
_PENDING_CONTENT_KEY = "add_note_pending_content"
_CONTENT_GENERATION_REQUEST_KEY = "add_note_content_generation_request"
_GENERATING_CONTENT_TEXT = "Генерирую содержание заметки..."
_CONTENT_FULLNESS_KEY = "add_note_content_fullness"


def render_add_note_page(
    note_command_service: NoteCommandService,
    metadata_suggester: NoteMetadataSuggester,
    content_generator: NoteContentGenerator,
) -> None:
    _apply_pending_content()
    _apply_pending_suggestion()

    with st.form("create-note"):
        title = st.text_input(
            "Название",
            key=_TITLE_KEY,
        )
        content_submitted = st.form_submit_button("Предложить содержание")
        content_fullness = st.slider(
            "Fullness генерируемого содержания",
            min_value=4,
            max_value=10,
            value=7,
            key=_CONTENT_FULLNESS_KEY,
        )
        markdown_content = st.text_area(
            "Markdown",
            key=_MARKDOWN_KEY,
            height=500,
            placeholder="Содержание заметки",
        )
        suggestion_submitted = st.form_submit_button("Предложить метаданные")

        group = st.text_input(
            "Группа",
            key=_GROUP_KEY,
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
        knowledge = st.slider(
            "Знание",
            min_value=0,
            max_value=10,
            value=0,
        )
        automatic_fullness = st.toggle(
            "Рассчитать заполненность автоматически",
            value=True,
        )

        if automatic_fullness:
            fullness = estimate_note_fullness(markdown_content)
            st.caption(f"Заполненность: {fullness}")
        else:
            fullness = st.slider(
                "Заполненность",
                min_value=0,
                max_value=10,
                value=5,
            )

        submitted = st.form_submit_button(
            "Создать заметку",
            type="primary",
        )

    if suggestion_submitted:
        _suggest_metadata(
            metadata_suggester,
            markdown_content,
        )

        return

    if content_submitted:
        normalized_title = title.strip()

        if not normalized_title:
            st.error("Сначала укажите название заметки.")

            return

        st.session_state[_PENDING_CONTENT_KEY] = _GENERATING_CONTENT_TEXT
        st.session_state[_CONTENT_GENERATION_REQUEST_KEY] = (
            normalized_title,
            content_fullness,
        )
        st.rerun()

    generation_request = st.session_state.pop(_CONTENT_GENERATION_REQUEST_KEY, None)

    if generation_request is not None:
        requested_title, requested_fullness = generation_request
        _generate_content(
            content_generator,
            requested_title,
            requested_fullness,
        )

        return

    suggestion = _get_cached_suggestion(markdown_content.strip())

    if suggestion is not None:
        st.caption(f"Ключевые понятия: {', '.join(suggestion.key_concepts)}")

    if not submitted:
        return

    if not title.strip() or not markdown_content.strip():
        st.error("Название и содержимое заметки обязательны.")

        return

    command = CreateNoteCommand(
        title=title.strip(),
        group=group.strip(),
        comment=comment.strip(),
        importance=importance,
        knowledge=knowledge,
        fullness=(estimate_note_fullness(markdown_content) if automatic_fullness else fullness),
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
        except (HTTPError, RuntimeError, ValidationError) as error:
            st.error(f"Не удалось предложить метаданные: {error}")

            return

        st.session_state[_SUGGESTION_KEY] = suggestion
        st.session_state[_SUGGESTION_SOURCE_KEY] = normalized_markdown
        st.session_state[_PENDING_SUGGESTION_KEY] = True

        st.rerun()


def _get_cached_suggestion(
    markdown_content: str,
) -> NoteMetadataSuggestion | None:
    if st.session_state.get(_SUGGESTION_SOURCE_KEY) != markdown_content:
        return None

    return st.session_state.get(_SUGGESTION_KEY)


def _apply_pending_suggestion() -> None:
    if not st.session_state.pop(_PENDING_SUGGESTION_KEY, False):
        return

    suggestion = st.session_state.get(_SUGGESTION_KEY)

    if suggestion is None:
        return

    st.session_state[_TITLE_KEY] = suggestion.title
    st.session_state[_GROUP_KEY] = suggestion.group
    st.session_state[_COMMENT_KEY] = suggestion.comment


def _generate_content(
    content_generator: NoteContentGenerator,
    title: str,
    fullness: int,
) -> None:
    try:
        with st.spinner("Генерирую содержание заметки..."):
            generated_content = content_generator.generate(
                title,
                fullness=fullness,
            )
    except (HTTPError, RuntimeError, ValidationError) as error:
        st.error(f"Не удалось предложить содержание: {error}")

        return

    st.session_state[_PENDING_CONTENT_KEY] = generated_content
    st.rerun()


def _apply_pending_content() -> None:
    generated_content = st.session_state.pop(_PENDING_CONTENT_KEY, None)

    if generated_content is None:
        return

    st.session_state[_MARKDOWN_KEY] = generated_content

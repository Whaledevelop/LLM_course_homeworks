import streamlit as st
from httpx import HTTPError
from collections.abc import Callable
from uuid import UUID

from tutor_bot.application.delete_note_command import DeleteNoteCommand
from tutor_bot.application.note_command_service import NoteCommandService
from tutor_bot.application.note_details import NoteDetails
from tutor_bot.application.note_fullness import estimate_note_fullness
from tutor_bot.application.update_note_command import UpdateNoteCommand
from tutor_bot.generation.note_content_generator import NoteContentGenerator


_ACTIVE_EDITOR_KEY = "active_note_editor"
_ACTIVE_DELETE_KEY = "active_note_delete"
_PENDING_CONTENT_PREFIX = "pending_note_content_"
_EDITOR_MARKDOWN_PREFIX = "edit_note_markdown_"


def render_note_actions(
    note_details: NoteDetails,
    note_command_service: NoteCommandService,
    test_note: Callable[[UUID], None],
    content_generator: NoteContentGenerator,
) -> str | None:
    note_id = str(note_details.id)
    columns_count = 4 if note_details.fullness < 4 else 3
    action_columns = st.columns(columns_count)
    edit_column = action_columns[0]
    test_column = action_columns[1]
    delete_column = action_columns[-1]

    edit_requested = edit_column.button(
        "Редактировать заметку",
        key=f"open-editor-{note_id}",
        type="primary",
        width="stretch",
    )

    test_column.button(
        "Тестировать заметку",
        key=f"test-note-{note_id}",
        type="primary",
        width="stretch",
        on_click=test_note,
        args=(note_details.id,),
    )

    delete_requested = delete_column.button(
        "Удалить заметку",
        key=f"open-delete-{note_id}",
        type="tertiary",
        width="stretch",
    )

    generate_requested = False
    save_requested = False

    if note_details.fullness < 4:
        generate_requested = action_columns[2].button(
            "Сгенерировать содержание",
            key=f"generate-content-{note_id}",
            width="stretch",
        )

    if edit_requested:
        active_editor = st.session_state.get(_ACTIVE_EDITOR_KEY)
        save_requested = active_editor == note_id

        if active_editor != note_id:
            st.session_state[_ACTIVE_EDITOR_KEY] = note_id

        st.session_state.pop(_ACTIVE_DELETE_KEY, None)

    if delete_requested:
        st.session_state[_ACTIVE_DELETE_KEY] = note_id
        st.session_state.pop(_ACTIVE_EDITOR_KEY, None)

    if generate_requested:
        try:
            with st.spinner("Расширяю содержание заметки..."):
                generated_content = content_generator.generate(
                    note_details.title,
                    note_details.markdown_content,
                )
        except (HTTPError, RuntimeError) as error:
            st.error(f"Не удалось сгенерировать содержание: {error}")

            return None

        st.session_state[f"{_PENDING_CONTENT_PREFIX}{note_details.id}"] = generated_content
        st.session_state[_ACTIVE_EDITOR_KEY] = note_id
        st.session_state.pop(_ACTIVE_DELETE_KEY, None)
        st.rerun()

    if st.session_state.get(_ACTIVE_EDITOR_KEY) == note_id:
        return _render_edit_form(
            note_details,
            note_command_service,
            content_generator,
            save_requested,
        )

    if st.session_state.get(_ACTIVE_DELETE_KEY) == note_id:
        return _render_delete_confirmation(
            note_details,
            note_command_service,
        )

    return None


def _render_edit_form(
    note_details: NoteDetails,
    note_command_service: NoteCommandService,
    content_generator: NoteContentGenerator,
    save_requested: bool,
) -> str | None:
    pending_content_key = f"{_PENDING_CONTENT_PREFIX}{note_details.id}"
    editor_markdown_key = f"{_EDITOR_MARKDOWN_PREFIX}{note_details.id}"
    pending_content = st.session_state.pop(pending_content_key, None)

    if pending_content is not None:
        st.session_state[editor_markdown_key] = pending_content
    elif editor_markdown_key not in st.session_state:
        st.session_state[editor_markdown_key] = note_details.markdown_content

    title = st.text_input(
        "Название",
        value=note_details.title,
    )

    group = st.text_input(
        "Группа",
        value=note_details.group,
    )

    comment = st.text_input(
        "Комментарий",
        value=note_details.comment,
    )

    questions_for_tests_text = st.text_area(
        "Вопросы для тестов",
        value="\n".join(note_details.questions_for_tests),
        placeholder="Один вопрос на строку",
    )

    importance = st.slider(
        "Важность",
        min_value=0,
        max_value=10,
        value=note_details.importance,
    )

    knowledge = st.slider(
        "Знание",
        min_value=0,
        max_value=10,
        value=note_details.knowledge,
    )

    markdown_content = st.text_area(
        "Markdown",
        key=editor_markdown_key,
        height=500,
    )

    generate_submitted = False

    if note_details.fullness < 4:
        generate_submitted = st.button(
            "Сгенерировать содержание",
            key=f"generate-editor-content-{note_details.id}",
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
            value=note_details.fullness,
        )

    save_column, cancel_column = st.columns(2)

    submitted = save_column.button(
        "Сохранить",
        type="primary",
        key=f"save-note-{note_details.id}",
        width="stretch",
    )

    cancelled = cancel_column.button(
        "Отмена",
        key=f"cancel-edit-note-{note_details.id}",
        width="stretch",
    )

    if cancelled:
        _close_actions(note_details.id)
        st.rerun()

    if generate_submitted:
        try:
            with st.spinner("Расширяю содержание заметки..."):
                generated_content = content_generator.generate(
                    title.strip(),
                    markdown_content,
                )
        except (HTTPError, RuntimeError) as error:
            st.error(f"Не удалось сгенерировать содержание: {error}")

            return None

        st.session_state[pending_content_key] = generated_content
        st.rerun()

    if not submitted and not save_requested:
        return None

    command = UpdateNoteCommand(
        note_id=note_details.id,
        title=title,
        group=group,
        comment=comment,
        questions_for_tests=_parse_questions_for_tests(questions_for_tests_text),
        importance=importance,
        knowledge=knowledge,
        fullness=(estimate_note_fullness(markdown_content) if automatic_fullness else fullness),
        markdown_content=markdown_content,
    )

    updated_note = note_command_service.update_note(command)
    _close_actions(note_details.id)

    return f"Заметка «{updated_note.title}» сохранена."


def _parse_questions_for_tests(value: str) -> tuple[str, ...]:
    return tuple(question.strip() for question in value.splitlines() if question.strip())


def _render_delete_confirmation(
    note_details: NoteDetails,
    note_command_service: NoteCommandService,
) -> str | None:
    st.error(f"Удалить заметку «{note_details.title}»? Markdown и metadata-запись будут удалены.")

    confirm_column, cancel_column = st.columns(2)

    confirmed = confirm_column.button(
        "Подтвердить удаление",
        type="primary",
        key=f"confirm-delete-{note_details.id}",
        width="stretch",
    )

    cancelled = cancel_column.button(
        "Отмена",
        key=f"cancel-delete-{note_details.id}",
        width="stretch",
    )

    if cancelled:
        _close_actions(note_details.id)
        st.rerun()

    if not confirmed:
        return None

    deleted_note = note_command_service.delete_note(DeleteNoteCommand(note_id=note_details.id))
    _close_actions(note_details.id)

    return f"Заметка «{deleted_note.title}» удалена."


def _close_actions(note_id: UUID) -> None:
    st.session_state.pop(_ACTIVE_EDITOR_KEY, None)
    st.session_state.pop(_ACTIVE_DELETE_KEY, None)
    st.session_state.pop(f"{_EDITOR_MARKDOWN_PREFIX}{note_id}", None)
    st.session_state.pop(f"{_PENDING_CONTENT_PREFIX}{note_id}", None)

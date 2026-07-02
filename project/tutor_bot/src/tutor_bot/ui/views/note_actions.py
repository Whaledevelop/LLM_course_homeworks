import streamlit as st

from tutor_bot.application.delete_note_command import DeleteNoteCommand
from tutor_bot.application.note_command_service import NoteCommandService
from tutor_bot.application.note_details import NoteDetails
from tutor_bot.application.update_note_command import UpdateNoteCommand


_ACTIVE_EDITOR_KEY = "active_note_editor"
_ACTIVE_DELETE_KEY = "active_note_delete"


def render_note_actions(
    note_details: NoteDetails,
    note_command_service: NoteCommandService,
) -> str | None:
    note_id = str(note_details.id)
    edit_column, delete_column = st.columns(2)

    edit_requested = edit_column.button(
        "✎",
        help="Редактировать заметку",
        key=f"open-editor-{note_id}",
        width="stretch",
    )

    delete_requested = delete_column.button(
        "🗑",
        help="Удалить заметку",
        key=f"open-delete-{note_id}",
        width="stretch",
    )

    if edit_requested:
        st.session_state[_ACTIVE_EDITOR_KEY] = note_id
        st.session_state.pop(_ACTIVE_DELETE_KEY, None)

    if delete_requested:
        st.session_state[_ACTIVE_DELETE_KEY] = note_id
        st.session_state.pop(_ACTIVE_EDITOR_KEY, None)

    if st.session_state.get(_ACTIVE_EDITOR_KEY) == note_id:
        return _render_edit_form(
            note_details,
            note_command_service,
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
) -> str | None:
    with st.form(f"edit-note-{note_details.id}"):
        title = st.text_input(
            "Название",
            value=note_details.title,
        )

        theme = st.text_input(
            "Тема",
            value=note_details.theme,
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

        knowledge = st.slider(
            "Знание",
            min_value=0,
            max_value=10,
            value=note_details.knowledge,
        )

        markdown_content = st.text_area(
            "Markdown",
            value=note_details.markdown_content,
            height=500,
        )

        save_column, cancel_column = st.columns(2)

        submitted = save_column.form_submit_button(
            "Сохранить",
            type="primary",
            width="stretch",
        )

        cancelled = cancel_column.form_submit_button(
            "Отмена",
            width="stretch",
        )

    if cancelled:
        _close_actions()
        st.rerun()

    if not submitted:
        return None

    command = UpdateNoteCommand(
        note_id=note_details.id,
        title=title,
        theme=theme,
        comment=comment,
        importance=importance,
        knowledge=knowledge,
        markdown_content=markdown_content,
    )

    updated_note = note_command_service.update_note(command)
    _close_actions()

    return f"Заметка «{updated_note.title}» сохранена."


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
        _close_actions()
        st.rerun()

    if not confirmed:
        return None

    deleted_note = note_command_service.delete_note(DeleteNoteCommand(note_id=note_details.id))
    _close_actions()

    return f"Заметка «{deleted_note.title}» удалена."


def _close_actions() -> None:
    st.session_state.pop(_ACTIVE_EDITOR_KEY, None)
    st.session_state.pop(_ACTIVE_DELETE_KEY, None)

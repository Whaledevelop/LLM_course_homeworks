from uuid import uuid4

from tutor_bot.application.note_list_item import NoteListItem
from tutor_bot.ui.views.browse_notes_page import _sort_notes


def test_sort_notes_by_importance_uses_title_as_tie_breaker() -> None:
    notes = [
        _create_note("Бета", 7),
        _create_note("Гамма", 9),
        _create_note("Альфа", 7),
    ]

    sorted_notes = _sort_notes(notes, "По важности")

    assert [note.title for note in sorted_notes] == ["Гамма", "Альфа", "Бета"]


def test_sort_notes_alphabetically_ignores_importance() -> None:
    notes = [
        _create_note("Бета", 10),
        _create_note("Альфа", 1),
        _create_note("Гамма", 5),
    ]

    sorted_notes = _sort_notes(notes, "По алфавиту")

    assert [note.title for note in sorted_notes] == ["Альфа", "Бета", "Гамма"]


def _create_note(title: str, importance: int) -> NoteListItem:
    return NoteListItem(
        id=uuid4(),
        title=title,
        group="",
        importance=importance,
        knowledge=0,
    )

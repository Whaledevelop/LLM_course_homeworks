from uuid import uuid4

from tutor_bot.application.note_list_item import NoteListItem
from tutor_bot.ui.views import browse_notes_page
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


def test_sort_notes_by_importance_places_favorites_first() -> None:
    notes = [
        _create_note("Обычная важная", 10),
        _create_note("Избранная", 1, favorite=True),
        _create_note("Обычная", 5),
    ]

    sorted_notes = _sort_notes(notes, "По важности")

    assert [note.title for note in sorted_notes] == [
        "Избранная",
        "Обычная важная",
        "Обычная",
    ]


def test_sort_notes_alphabetically_places_favorites_first() -> None:
    notes = [
        _create_note("Альфа", 10),
        _create_note("Бета", 1, favorite=True),
    ]

    sorted_notes = _sort_notes(notes, "По алфавиту")

    assert [note.title for note in sorted_notes] == ["Бета", "Альфа"]


def test_scroll_to_page_top_consumes_request(monkeypatch) -> None:
    session_state = {"browse_notes_scroll_to_top": True}
    rendered_html = []
    monkeypatch.setattr(browse_notes_page.st, "session_state", session_state)
    monkeypatch.setattr(
        browse_notes_page.components,
        "html",
        lambda html, height: rendered_html.append((html, height)),
    )

    browse_notes_page._scroll_to_page_top_if_requested()

    assert "scrollTo" in rendered_html[0][0]
    assert rendered_html[0][1] == 0
    assert session_state == {}


def test_scroll_to_page_top_without_request_does_not_render(monkeypatch) -> None:
    rendered_html = []
    monkeypatch.setattr(browse_notes_page.st, "session_state", {})
    monkeypatch.setattr(
        browse_notes_page.components,
        "html",
        lambda html, height: rendered_html.append((html, height)),
    )

    browse_notes_page._scroll_to_page_top_if_requested()

    assert rendered_html == []


def _create_note(title: str, importance: int, favorite: bool = False) -> NoteListItem:
    return NoteListItem(
        id=uuid4(),
        title=title,
        group="",
        importance=importance,
        knowledge=0,
        favorite=favorite,
    )

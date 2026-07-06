from tutor_bot.application.note_metadata_suggestion import NoteMetadataSuggestion
from tutor_bot.ui.views import add_note_page


def test_metadata_suggestion_does_not_replace_entered_title(monkeypatch):
    suggestion = NoteMetadataSuggestion(
        title="Название от модели",
        group="Unity",
        questions_for_tests=(
            "Вопрос 1?",
            "Вопрос 2?",
            "Вопрос 3?",
            "Вопрос 4?",
            "Вопрос 5?",
        ),
        importance=7,
        key_concepts=("Memory",),
    )
    session_state = {
        add_note_page._TITLE_KEY: "Название пользователя",
        add_note_page._SUGGESTION_KEY: suggestion,
        add_note_page._PENDING_SUGGESTION_KEY: True,
    }
    monkeypatch.setattr(add_note_page.st, "session_state", session_state)

    add_note_page._apply_pending_suggestion()

    assert session_state[add_note_page._TITLE_KEY] == "Название пользователя"

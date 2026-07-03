from tutor_bot.application.note_fullness import estimate_note_fullness


def test_estimates_empty_and_one_page_notes() -> None:
    assert estimate_note_fullness("") == 0
    assert (
        estimate_note_fullness(
            "---\ntutor_bot_note_id: 0b2c61a8-505a-552c-b731-7b9627970eff\n---\n"
        )
        == 0
    )
    assert estimate_note_fullness("x" * 2500) == 5
    assert estimate_note_fullness("x" * 3000) == 6


def test_caps_fullness_at_ten() -> None:
    assert estimate_note_fullness("x" * 10000) == 10

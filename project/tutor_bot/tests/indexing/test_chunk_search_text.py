from tutor_bot.indexing.chunk_search_text import build_chunk_search_text


def test_build_chunk_search_text_includes_note_and_section_titles() -> None:
    search_text = build_chunk_search_text(
        "Основные решения по мультиплееру",
        "Netcode for GameObjects",
        "Официальный сетевой фреймворк Unity.",
    )

    assert search_text == (
        "Основные решения по мультиплееру\n\n"
        "Netcode for GameObjects\n\n"
        "Официальный сетевой фреймворк Unity."
    )

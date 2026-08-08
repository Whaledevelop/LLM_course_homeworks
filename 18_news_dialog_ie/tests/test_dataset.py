from dataset import is_english_text, is_news_dialog, unique_dialogs
from schemas import NewsDialog


def test_news_filter_requires_news_marker() -> None:
    assert is_news_dialog("Reuters reported a market event")
    assert not is_news_dialog("Write a poem about a garden")


def test_english_filter_rejects_non_english_text() -> None:
    assert is_english_text("BBC reported an election result in London")
    assert not is_english_text("Новости сообщили о выборах в Москве")


def test_unique_dialogs_removes_duplicate_ids_and_texts() -> None:
    dialogs = [
        NewsDialog("1", "source", "Reuters reported news"),
        NewsDialog("2", "source", " Reuters   reported NEWS "),
        NewsDialog("1", "source", "BBC reported news"),
    ]

    assert len(unique_dialogs(dialogs)) == 1

from tutor_bot.indexing.markdown_cleaner import MarkdownCleaner


def test_normalizes_markdown_whitespace() -> None:
    source = "\ufeff# Title  \r\n\r\n\r\nText \t \r\n\r\n"

    cleaned = MarkdownCleaner().clean(source)

    assert cleaned == "# Title  \n\nText\n"


def test_preserves_markdown_structure() -> None:
    source = "## Section\n\n- First item\n- [Second item](https://example.com)\n"

    cleaned = MarkdownCleaner().clean(source)

    assert cleaned == source


def test_preserves_fenced_code_content() -> None:
    source = "Before\n\n\n```csharp   \nvar value = 1;  \n\n\n```   \n\n\nAfter\n"

    cleaned = MarkdownCleaner().clean(source)

    assert cleaned == ("Before\n\n```csharp\nvar value = 1;  \n\n\n```\n\nAfter\n")


def test_cleaning_is_idempotent() -> None:
    cleaner = MarkdownCleaner()
    source = "# Title\r\n\r\n\r\nContent   \r\n"

    cleaned = cleaner.clean(source)

    assert cleaner.clean(cleaned) == cleaned

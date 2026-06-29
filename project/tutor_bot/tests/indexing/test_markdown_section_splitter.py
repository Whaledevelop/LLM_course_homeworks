from tutor_bot.indexing.markdown_section import MarkdownSection
from tutor_bot.indexing.markdown_section_splitter import (
    MarkdownSectionSplitter,
)


def test_splits_preamble_and_nested_sections() -> None:
    markdown = "Introduction\n# Root\nRoot content\n## Child\nChild content\n"

    sections = MarkdownSectionSplitter().split(markdown)

    assert sections == [
        MarkdownSection(
            heading_level=0,
            heading_title="",
            heading_path=(),
            content="Introduction\n",
        ),
        MarkdownSection(
            heading_level=1,
            heading_title="Root",
            heading_path=("Root",),
            content="Root content\n",
        ),
        MarkdownSection(
            heading_level=2,
            heading_title="Child",
            heading_path=("Root", "Child"),
            content="Child content\n",
        ),
    ]


def test_updates_hierarchy_for_sibling_sections() -> None:
    markdown = (
        "# Root\n## First\nFirst content\n### Deep\nDeep content\n## Second ##\nSecond content\n"
    )

    sections = MarkdownSectionSplitter().split(markdown)

    assert sections == [
        MarkdownSection(
            heading_level=2,
            heading_title="First",
            heading_path=("Root", "First"),
            content="First content\n",
        ),
        MarkdownSection(
            heading_level=3,
            heading_title="Deep",
            heading_path=("Root", "First", "Deep"),
            content="Deep content\n",
        ),
        MarkdownSection(
            heading_level=2,
            heading_title="Second",
            heading_path=("Root", "Second"),
            content="Second content\n",
        ),
    ]


def test_does_not_split_headings_inside_fenced_code() -> None:
    markdown = (
        "# Real section\n"
        "Before code\n"
        "```markdown\n"
        "# Not a section\n"
        "```\n"
        "After code\n"
        "## Next section\n"
        "Next content\n"
    )

    sections = MarkdownSectionSplitter().split(markdown)

    assert sections == [
        MarkdownSection(
            heading_level=1,
            heading_title="Real section",
            heading_path=("Real section",),
            content=("Before code\n```markdown\n# Not a section\n```\nAfter code\n"),
        ),
        MarkdownSection(
            heading_level=2,
            heading_title="Next section",
            heading_path=(
                "Real section",
                "Next section",
            ),
            content="Next content\n",
        ),
    ]


def test_keeps_empty_parent_headings_in_path() -> None:
    markdown = "# Root\n## Empty parent\n### Child\nChild content\n"

    sections = MarkdownSectionSplitter().split(markdown)

    assert sections == [
        MarkdownSection(
            heading_level=3,
            heading_title="Child",
            heading_path=(
                "Root",
                "Empty parent",
                "Child",
            ),
            content="Child content\n",
        )
    ]


def test_creates_preamble_section_without_headings() -> None:
    markdown = "First paragraph\n\nSecond paragraph\n"

    sections = MarkdownSectionSplitter().split(markdown)

    assert sections == [
        MarkdownSection(
            heading_level=0,
            heading_title="",
            heading_path=(),
            content="First paragraph\n\nSecond paragraph\n",
        )
    ]

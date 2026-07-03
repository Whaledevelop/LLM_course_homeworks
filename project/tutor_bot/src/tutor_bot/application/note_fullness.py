from math import ceil
import re


_CHARACTERS_PER_FULLNESS_POINT = 500
_FRONTMATTER_PATTERN = re.compile(
    r"\A---[ \t]*\r?\n.*?\r?\n---[ \t]*(?:\r?\n|\Z)",
    re.DOTALL,
)


def estimate_note_fullness(markdown_content: str) -> int:
    content = _FRONTMATTER_PATTERN.sub("", markdown_content, count=1)
    content_length = len(content.strip())

    if content_length == 0:
        return 0

    return min(10, ceil(content_length / _CHARACTERS_PER_FULLNESS_POINT))

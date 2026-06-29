import re

from tutor_bot.indexing.markdown_section import MarkdownSection


_FENCE_PATTERN = re.compile(r"^\s*(?P<fence>`{3,}|~{3,})")
_HEADING_PATTERN = re.compile(r"^(?P<marks>#{1,6})(?:[ \t]+(?P<title>.*))?[ \t]*$")
_CLOSING_MARKS_PATTERN = re.compile(r"[ \t]+#+[ \t]*$")


class MarkdownSectionSplitter:
    def split(
        self,
        markdown_content: str,
    ) -> list[MarkdownSection]:
        sections = []
        heading_stack: list[tuple[int, str]] = []
        content_lines = []
        current_heading_level = 0
        current_heading_title = ""
        fence_marker = ""
        fence_length = 0

        for line in markdown_content.splitlines():
            if fence_marker:
                content_lines.append(line)

                if self._is_closing_fence(
                    line,
                    fence_marker,
                    fence_length,
                ):
                    fence_marker = ""
                    fence_length = 0

                continue

            opening_fence = _FENCE_PATTERN.match(line)

            if opening_fence:
                fence = opening_fence.group("fence")
                fence_marker = fence[0]
                fence_length = len(fence)
                content_lines.append(line)
                continue

            heading_match = _HEADING_PATTERN.match(line)

            if not heading_match:
                content_lines.append(line)
                continue

            self._append_section(
                sections,
                content_lines,
                current_heading_level,
                current_heading_title,
                heading_stack,
            )

            heading_level = len(heading_match.group("marks"))
            heading_title = self._normalize_heading_title(heading_match.group("title") or "")

            while heading_stack and heading_stack[-1][0] >= heading_level:
                heading_stack.pop()

            heading_stack.append(
                (
                    heading_level,
                    heading_title,
                )
            )

            current_heading_level = heading_level
            current_heading_title = heading_title
            content_lines = []

        self._append_section(
            sections,
            content_lines,
            current_heading_level,
            current_heading_title,
            heading_stack,
        )

        return sections

    def _append_section(
        self,
        sections: list[MarkdownSection],
        content_lines: list[str],
        heading_level: int,
        heading_title: str,
        heading_stack: list[tuple[int, str]],
    ) -> None:
        content = "\n".join(content_lines).strip("\n")

        if not content.strip():
            return

        sections.append(
            MarkdownSection(
                heading_level=heading_level,
                heading_title=heading_title,
                heading_path=tuple(title for _, title in heading_stack),
                content=content + "\n",
            )
        )

    def _normalize_heading_title(
        self,
        heading_title: str,
    ) -> str:
        return _CLOSING_MARKS_PATTERN.sub(
            "",
            heading_title,
        ).strip()

    def _is_closing_fence(
        self,
        line: str,
        fence_marker: str,
        fence_length: int,
    ) -> bool:
        stripped_line = line.strip()

        return len(stripped_line) >= fence_length and set(stripped_line) == {fence_marker}

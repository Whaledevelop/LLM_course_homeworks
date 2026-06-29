import re


_FENCE_PATTERN = re.compile(r"^\s*(?P<fence>`{3,}|~{3,})")


class MarkdownCleaner:
    def clean(
        self,
        markdown_content: str,
    ) -> str:
        normalized_content = (
            markdown_content.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
        )

        cleaned_lines = []
        pending_blank_line = False
        fence_marker = ""
        fence_length = 0

        for source_line in normalized_content.split("\n"):
            if fence_marker:
                is_closing_fence = self._is_closing_fence(
                    source_line,
                    fence_marker,
                    fence_length,
                )

                cleaned_lines.append(source_line.rstrip() if is_closing_fence else source_line)

                if is_closing_fence:
                    fence_marker = ""
                    fence_length = 0

                continue

            opening_fence = _FENCE_PATTERN.match(source_line)

            if opening_fence:
                if pending_blank_line and cleaned_lines:
                    cleaned_lines.append("")

                fence = opening_fence.group("fence")
                fence_marker = fence[0]
                fence_length = len(fence)
                pending_blank_line = False
                cleaned_lines.append(source_line.rstrip())
                continue

            cleaned_line = self._clean_content_line(source_line)

            if not cleaned_line:
                pending_blank_line = True
                continue

            if pending_blank_line and cleaned_lines:
                cleaned_lines.append("")

            pending_blank_line = False
            cleaned_lines.append(cleaned_line)

        if not cleaned_lines:
            return ""

        return "\n".join(cleaned_lines).rstrip("\n") + "\n"

    def _clean_content_line(
        self,
        source_line: str,
    ) -> str:
        if not source_line.strip():
            return ""

        cleaned_line = source_line.rstrip()

        if source_line.endswith("  "):
            return cleaned_line + "  "

        return cleaned_line

    def _is_closing_fence(
        self,
        source_line: str,
        fence_marker: str,
        fence_length: int,
    ) -> bool:
        stripped_line = source_line.strip()

        return len(stripped_line) >= fence_length and set(stripped_line) == {fence_marker}

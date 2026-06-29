from tutor_bot.indexing.markdown_chunk import MarkdownChunk
from tutor_bot.indexing.markdown_section import MarkdownSection


class MarkdownSectionChunker:
    def __init__(
        self,
        max_chars: int = 800,
        overlap_chars: int = 100,
    ) -> None:
        if max_chars <= 0:
            raise ValueError("Maximum chunk size must be positive")

        if overlap_chars < 0 or overlap_chars >= max_chars:
            raise ValueError("Chunk overlap must be non-negative and smaller than chunk size")

        self._max_chars = max_chars
        self._overlap_chars = overlap_chars

    def chunk(
        self,
        sections: list[MarkdownSection],
    ) -> list[MarkdownChunk]:
        chunks = []

        for section_index, section in enumerate(sections):
            section_contents = self._split_content(section.content)

            for chunk_index, content in enumerate(section_contents):
                chunks.append(
                    MarkdownChunk(
                        section_index=section_index,
                        chunk_index=chunk_index,
                        heading_level=section.heading_level,
                        heading_title=section.heading_title,
                        heading_path=section.heading_path,
                        content=content,
                    )
                )

        return chunks

    def _split_content(
        self,
        content: str,
    ) -> list[str]:
        normalized_content = content.strip("\n")

        if len(normalized_content) <= self._max_chars:
            return [normalized_content]

        chunks = []
        chunk_start = 0

        while chunk_start < len(normalized_content):
            maximum_end = min(
                chunk_start + self._max_chars,
                len(normalized_content),
            )

            chunk_end = self._find_chunk_end(
                normalized_content,
                chunk_start,
                maximum_end,
            )

            chunk_content = normalized_content[chunk_start:chunk_end].strip("\n")

            if chunk_content.strip():
                chunks.append(chunk_content)

            if chunk_end >= len(normalized_content):
                break

            next_start = self._find_next_start(
                normalized_content,
                chunk_start,
                chunk_end,
            )

            if next_start <= chunk_start:
                next_start = chunk_end

            chunk_start = next_start

        return chunks

    def _find_chunk_end(
        self,
        content: str,
        chunk_start: int,
        maximum_end: int,
    ) -> int:
        if maximum_end >= len(content):
            return maximum_end

        minimum_end = chunk_start + self._max_chars // 2

        for separator in (
            "\n\n",
            "\n",
            ". ",
            " ",
        ):
            separator_index = content.rfind(
                separator,
                minimum_end,
                maximum_end,
            )

            if separator_index >= minimum_end:
                return separator_index + len(separator)

        return maximum_end

    def _find_next_start(
        self,
        content: str,
        chunk_start: int,
        chunk_end: int,
    ) -> int:
        overlap_start = max(
            chunk_end - self._overlap_chars,
            chunk_start + 1,
        )

        whitespace_index = content.find(
            " ",
            overlap_start,
            chunk_end,
        )

        if whitespace_index == -1:
            return overlap_start

        return whitespace_index + 1

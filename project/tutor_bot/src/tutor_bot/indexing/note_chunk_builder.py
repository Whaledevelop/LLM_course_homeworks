from datetime import datetime
from uuid import UUID, uuid5

from tutor_bot.indexing.indexed_chunk import IndexedChunk
from tutor_bot.indexing.markdown_cleaner import MarkdownCleaner
from tutor_bot.indexing.markdown_section_chunker import (
    MarkdownSectionChunker,
)
from tutor_bot.indexing.markdown_section_splitter import (
    MarkdownSectionSplitter,
)
from tutor_bot.schemas.note_metadata import NoteMetadata


class NoteChunkBuilder:
    def __init__(
        self,
        markdown_cleaner: MarkdownCleaner,
        section_splitter: MarkdownSectionSplitter,
        section_chunker: MarkdownSectionChunker,
    ) -> None:
        self._markdown_cleaner = markdown_cleaner
        self._section_splitter = section_splitter
        self._section_chunker = section_chunker

    def build(
        self,
        note_id: UUID,
        metadata: NoteMetadata,
        markdown_content: str,
        source_modified_at: datetime,
    ) -> list[IndexedChunk]:
        cleaned_markdown = self._markdown_cleaner.clean(markdown_content)
        sections = self._section_splitter.split(cleaned_markdown)
        markdown_chunks = self._section_chunker.chunk(sections)
        indexed_chunks = []

        for markdown_chunk in markdown_chunks:
            heading_context = " > ".join(markdown_chunk.heading_path)

            text = markdown_chunk.content

            if heading_context:
                text = f"{heading_context}\n\n{text}"

            indexed_chunks.append(
                IndexedChunk(
                    chunk_id=uuid5(
                        note_id,
                        (f"{markdown_chunk.section_index}:{markdown_chunk.chunk_index}"),
                    ),
                    note_id=note_id,
                    section_index=markdown_chunk.section_index,
                    chunk_index=markdown_chunk.chunk_index,
                    note_title=metadata.last_recorded_name,
                    heading_title=markdown_chunk.heading_title,
                    heading_path=markdown_chunk.heading_path,
                    text=text,
                    theme=metadata.theme,
                    difficulty=metadata.difficulty,
                    importance=metadata.importance,
                    completeness=metadata.completeness,
                    mastery=metadata.mastery,
                    relative_path=metadata.relative_path,
                    source_modified_at=source_modified_at,
                )
            )

        return indexed_chunks

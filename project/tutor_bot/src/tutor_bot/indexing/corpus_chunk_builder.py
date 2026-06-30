from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from uuid import UUID

from tutor_bot.indexing.indexed_chunk import IndexedChunk
from tutor_bot.indexing.note_chunk_builder import NoteChunkBuilder
from tutor_bot.infrastructure.markdown_document import MarkdownDocument
from tutor_bot.infrastructure.notes_metadata_repository import (
    NotesMetadataRepository,
)
from tutor_bot.schemas.note_metadata import NoteMetadata


class CorpusChunkBuilder:
    def __init__(
        self,
        metadata_repository: NotesMetadataRepository,
        source_notes_dir: Path,
        note_chunk_builder: NoteChunkBuilder,
    ) -> None:
        self._metadata_repository = metadata_repository
        self._source_notes_dir = source_notes_dir.resolve()
        self._note_chunk_builder = note_chunk_builder

    def build(self) -> list[IndexedChunk]:
        catalog = self._metadata_repository.load()
        indexed_chunks = []

        sorted_notes = sorted(
            catalog.notes.items(),
            key=lambda item: (
                item[1].relative_path.as_posix().casefold(),
                str(item[0]),
            ),
        )

        for note_id, metadata in sorted_notes:
            indexed_chunks.extend(
                self._build_note_chunks(
                    note_id,
                    metadata,
                )
            )

        return indexed_chunks

    def _build_note_chunks(
        self,
        note_id: UUID,
        metadata: NoteMetadata,
    ) -> list[IndexedChunk]:
        note_path = self._resolve_path(metadata.relative_path)
        file_content = note_path.read_text(encoding="utf-8-sig")
        document = MarkdownDocument.parse(
            file_content,
            note_path,
        )

        if document.note_id != note_id:
            raise ValueError(f"Note id mismatch: expected {note_id}, found {document.note_id}")

        source_modified_at = datetime.fromtimestamp(
            note_path.stat().st_mtime,
            tz=timezone.utc,
        )

        return self._note_chunk_builder.build(
            note_id,
            metadata,
            document.content,
            source_modified_at,
        )

    def _resolve_path(
        self,
        relative_path: PurePosixPath,
    ) -> Path:
        note_path = (self._source_notes_dir / relative_path).resolve()

        if not note_path.is_relative_to(self._source_notes_dir):
            raise ValueError(f"Note path escapes source directory: {note_path}")

        if not note_path.is_file():
            raise FileNotFoundError(f"Note file not found: {note_path}")

        return note_path

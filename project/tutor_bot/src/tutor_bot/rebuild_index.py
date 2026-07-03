import json
from time import perf_counter

from tutor_bot.config import get_settings
from tutor_bot.indexing.bm25_chunk_index import Bm25ChunkIndex
from tutor_bot.indexing.chroma_chunk_index import (
    ChromaChunkIndex,
)
from tutor_bot.indexing.corpus_chunk_builder import (
    CorpusChunkBuilder,
)
from tutor_bot.indexing.full_reindex_service import (
    FullReindexService,
)
from tutor_bot.indexing.markdown_cleaner import MarkdownCleaner
from tutor_bot.indexing.markdown_section_chunker import (
    MarkdownSectionChunker,
)
from tutor_bot.indexing.markdown_section_splitter import (
    MarkdownSectionSplitter,
)
from tutor_bot.indexing.note_chunk_builder import NoteChunkBuilder
from tutor_bot.indexing.sentence_transformer_embedding_service import (
    SentenceTransformerEmbeddingService,
)
from tutor_bot.infrastructure.active_database_service import ActiveDatabaseService
from tutor_bot.infrastructure.database_notes_repository import DatabaseNotesRepository


def main() -> int:
    settings = get_settings()
    active_database = ActiveDatabaseService(settings.data_dir).get_active()

    if active_database is None:
        raise RuntimeError("Active DB is not selected")

    metadata_repository = DatabaseNotesRepository(
        settings.data_dir / "metadata",
        active_database.db_id,
        active_database.root_path,
    )

    note_chunk_builder = NoteChunkBuilder(
        MarkdownCleaner(),
        MarkdownSectionSplitter(),
        MarkdownSectionChunker(),
    )

    corpus_chunk_builder = CorpusChunkBuilder(
        metadata_repository,
        active_database.root_path,
        note_chunk_builder,
    )

    embedding_service = SentenceTransformerEmbeddingService(device="cpu")

    chroma_index = ChromaChunkIndex(active_database.indexes_dir / "chroma")
    bm25_index = Bm25ChunkIndex(active_database.indexes_dir / "bm25")

    reindex_service = FullReindexService(
        corpus_chunk_builder,
        embedding_service,
        chroma_index,
        bm25_index,
    )

    started_at = perf_counter()
    chunk_count = reindex_service.rebuild()
    elapsed_seconds = perf_counter() - started_at

    print(
        json.dumps(
            {
                "db_id": active_database.db_id,
                "notes_directory": str(active_database.root_path),
                "chunks": chunk_count,
                "chroma_records": chroma_index.count,
                "bm25_records": chunk_count,
                "embedding_dimension": (embedding_service.dimension),
                "elapsed_seconds": round(
                    elapsed_seconds,
                    3,
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from tutor_bot.indexing.bm25_chunk_index import Bm25ChunkIndex
from tutor_bot.indexing.chroma_chunk_index import (
    ChromaChunkIndex,
)
from tutor_bot.indexing.corpus_chunk_builder import (
    CorpusChunkBuilder,
)
from tutor_bot.indexing.embedding_service import EmbeddingService


class FullReindexService:
    def __init__(
        self,
        corpus_chunk_builder: CorpusChunkBuilder,
        embedding_service: EmbeddingService,
        chroma_index: ChromaChunkIndex,
        bm25_index: Bm25ChunkIndex,
    ) -> None:
        self._corpus_chunk_builder = corpus_chunk_builder
        self._embedding_service = embedding_service
        self._chroma_index = chroma_index
        self._bm25_index = bm25_index

    def rebuild(self) -> int:
        chunks = self._corpus_chunk_builder.build()

        if not chunks:
            self._chroma_index.replace_all([], [])
            self._bm25_index.rebuild([])

            return 0

        embeddings = self._embedding_service.embed_passages([chunk.text for chunk in chunks])

        self._chroma_index.replace_all(
            chunks,
            embeddings,
        )

        if self._chroma_index.count != len(chunks):
            raise RuntimeError("Chroma record count does not match corpus chunks")

        bm25_record_count = self._bm25_index.rebuild(chunks)

        if bm25_record_count != len(chunks):
            raise RuntimeError("BM25 record count does not match corpus chunks")

        return len(chunks)

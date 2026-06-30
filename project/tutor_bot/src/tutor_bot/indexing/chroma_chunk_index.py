from pathlib import Path, PurePosixPath
from uuid import UUID

import chromadb
from chromadb import Collection

from tutor_bot.indexing.indexed_chunk import IndexedChunk
from tutor_bot.retrieval.chunk_search_result import ChunkSearchResult


_DEFAULT_COLLECTION_NAME = "tutor_bot_chunks"


class ChromaChunkIndex:
    def __init__(
        self,
        index_directory: Path,
        collection_name: str = _DEFAULT_COLLECTION_NAME,
        batch_size: int = 128,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("Chroma batch size must be positive")

        index_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._client = chromadb.PersistentClient(path=str(index_directory))

        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            configuration={
                "hnsw": {
                    "space": "cosine",
                }
            },
        )

        self._batch_size = batch_size

    @property
    def count(self) -> int:
        return self._collection.count()

    def replace_all(
        self,
        chunks: list[IndexedChunk],
        embeddings: list[list[float]],
    ) -> None:
        self._validate_records(
            chunks,
            embeddings,
        )

        target_ids = {str(chunk.chunk_id) for chunk in chunks}

        self.upsert(
            chunks,
            embeddings,
        )

        existing_records = self._collection.get()
        stale_ids = [
            record_id for record_id in existing_records["ids"] if record_id not in target_ids
        ]

        if stale_ids:
            self._collection.delete(ids=stale_ids)

    def upsert(
        self,
        chunks: list[IndexedChunk],
        embeddings: list[list[float]],
    ) -> None:
        self._validate_records(
            chunks,
            embeddings,
        )

        for batch_start in range(
            0,
            len(chunks),
            self._batch_size,
        ):
            batch_end = batch_start + self._batch_size
            batch_chunks = chunks[batch_start:batch_end]
            batch_embeddings = embeddings[batch_start:batch_end]

            self._collection.upsert(
                ids=[str(chunk.chunk_id) for chunk in batch_chunks],
                embeddings=batch_embeddings,
                documents=[chunk.text for chunk in batch_chunks],
                metadatas=[self._create_metadata(chunk) for chunk in batch_chunks],
            )

    def delete_note(
        self,
        note_id: UUID,
    ) -> None:
        self._collection.delete(
            where={
                "note_id": str(note_id),
            }
        )

    def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        theme: str | None = None,
    ) -> list[ChunkSearchResult]:
        if limit <= 0:
            raise ValueError("Search result limit must be positive")

        if self.count == 0:
            return []

        where = None

        if theme:
            where = {
                "theme": theme,
            }

        query_result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(limit, self.count),
            where=where,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        ids = query_result["ids"][0]
        documents = query_result["documents"][0]
        metadatas = query_result["metadatas"][0]
        distances = query_result["distances"][0]
        results = []

        for chunk_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            heading_path = str(metadata["heading_path"])

            results.append(
                ChunkSearchResult(
                    chunk_id=UUID(chunk_id),
                    note_id=UUID(str(metadata["note_id"])),
                    note_title=str(metadata["note_title"]),
                    section_index=int(metadata["section_index"]),
                    chunk_index=int(metadata["chunk_index"]),
                    heading_title=str(metadata["heading_title"]),
                    heading_path=(tuple(heading_path.split(" > ")) if heading_path else ()),
                    text=str(document),
                    theme=str(metadata["theme"]),
                    difficulty=str(metadata["difficulty"]),
                    relative_path=PurePosixPath(str(metadata["relative_path"])),
                    score=1.0 - float(distance),
                    retrieval_method="vector",
                )
            )

        return results

    def _validate_records(
        self,
        chunks: list[IndexedChunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk and embedding counts must match")

        chunk_ids = [chunk.chunk_id for chunk in chunks]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Chunk ids must be unique")

        embedding_dimensions = {len(embedding) for embedding in embeddings}

        if len(embedding_dimensions) > 1:
            raise ValueError("Embedding dimensions must be consistent")

    def _create_metadata(
        self,
        chunk: IndexedChunk,
    ) -> dict[str, str | int]:
        return {
            "note_id": str(chunk.note_id),
            "note_title": chunk.note_title,
            "section_index": chunk.section_index,
            "chunk_index": chunk.chunk_index,
            "heading_title": chunk.heading_title,
            "heading_path": " > ".join(chunk.heading_path),
            "theme": chunk.theme,
            "difficulty": chunk.difficulty,
            "importance": chunk.importance,
            "completeness": chunk.completeness,
            "mastery": chunk.mastery,
            "relative_path": chunk.relative_path.as_posix(),
            "source_modified_at": (chunk.source_modified_at.isoformat()),
        }

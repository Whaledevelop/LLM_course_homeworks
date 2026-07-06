from pathlib import Path, PurePosixPath
from uuid import UUID

import bm25s
from bm25s.tokenization import Tokenizer

from tutor_bot.indexing.chunk_search_text import build_chunk_search_text
from tutor_bot.indexing.indexed_chunk import IndexedChunk
from tutor_bot.retrieval.chunk_search_result import ChunkSearchResult


_TOKEN_PATTERN = r"(?u)\b\w+\b"


class Bm25ChunkIndex:
    def __init__(
        self,
        index_directory: Path,
    ) -> None:
        self._index_directory = index_directory
        self._retriever: bm25s.BM25 | None = None
        self._tokenizer: Tokenizer | None = None

    def rebuild(
        self,
        chunks: list[IndexedChunk],
    ) -> int:
        if not chunks:
            raise ValueError("BM25 index requires at least one chunk")

        chunk_ids = [chunk.chunk_id for chunk in chunks]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Chunk ids must be unique")

        corpus = [self._create_record(chunk) for chunk in chunks]

        tokenizer = Tokenizer(
            splitter=_TOKEN_PATTERN,
            stopwords=[],
        )

        corpus_tokens = tokenizer.tokenize(
            [
                build_chunk_search_text(
                    chunk.note_title,
                    chunk.heading_title,
                    chunk.text,
                )
                for chunk in chunks
            ],
            show_progress=False,
            return_as="tuple",
        )

        retriever = bm25s.BM25(
            corpus=corpus,
        )

        retriever.index(
            corpus_tokens,
            show_progress=False,
        )

        self._index_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        retriever.save(
            str(self._index_directory),
            corpus=corpus,
            show_progress=False,
        )

        tokenizer.save_vocab(str(self._index_directory))

        self._retriever = retriever
        self._tokenizer = tokenizer

        return len(chunks)

    def search(
        self,
        query: str,
        limit: int = 10,
        group: str | None = None,
    ) -> list[ChunkSearchResult]:
        if limit <= 0:
            raise ValueError("Search result limit must be positive")

        retriever, tokenizer = self._load()
        corpus_count = len(retriever.corpus)

        if corpus_count == 0:
            return []

        query_tokens = tokenizer.tokenize(
            [query],
            update_vocab=False,
            show_progress=False,
        )

        candidate_limit = corpus_count if group else min(limit, corpus_count)

        records, scores = retriever.retrieve(
            query_tokens,
            k=candidate_limit,
            show_progress=False,
        )

        results = []

        for record, score in zip(records[0], scores[0]):
            if group and record["group"] != group:
                continue

            results.append(
                ChunkSearchResult(
                    chunk_id=UUID(str(record["chunk_id"])),
                    note_id=UUID(str(record["note_id"])),
                    note_title=str(record["note_title"]),
                    section_index=int(record["section_index"]),
                    chunk_index=int(record["chunk_index"]),
                    heading_title=str(record["heading_title"]),
                    heading_path=tuple(record["heading_path"]),
                    text=str(record["text"]),
                    group=str(record["group"]),
                    relative_path=PurePosixPath(str(record["relative_path"])),
                    score=float(score),
                    retrieval_method="bm25",
                )
            )

            if len(results) == limit:
                break

        return results

    def _load(self) -> tuple[bm25s.BM25, Tokenizer]:
        if self._retriever is None or self._tokenizer is None:
            if not self._index_directory.is_dir():
                raise FileNotFoundError(f"BM25 index not found: {self._index_directory}")

            self._retriever = bm25s.BM25.load(
                str(self._index_directory),
                load_corpus=True,
            )

            self._tokenizer = Tokenizer(
                splitter=_TOKEN_PATTERN,
                stopwords=[],
            )

            self._tokenizer.load_vocab(str(self._index_directory))

        return self._retriever, self._tokenizer

    def _create_record(
        self,
        chunk: IndexedChunk,
    ) -> dict[str, object]:
        return {
            "chunk_id": str(chunk.chunk_id),
            "note_id": str(chunk.note_id),
            "section_index": chunk.section_index,
            "chunk_index": chunk.chunk_index,
            "note_title": chunk.note_title,
            "heading_title": chunk.heading_title,
            "heading_path": list(chunk.heading_path),
            "text": chunk.text,
            "group": chunk.group,
            "importance": chunk.importance,
            "knowledge": chunk.knowledge,
            "relative_path": (chunk.relative_path.as_posix()),
            "source_modified_at": (chunk.source_modified_at.isoformat()),
        }

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb import Collection
from chromadb.errors import NotFoundError

from dataset import DocumentRecord
from embedding_service import HashingEmbeddingFunction


def create_client(path: Path) -> chromadb.PersistentClient:
    path.mkdir(parents=True, exist_ok=True)

    return chromadb.PersistentClient(path=str(path))


def recreate_collection(
    client: chromadb.PersistentClient,
    name: str,
    embedding_function: HashingEmbeddingFunction,
    space: str = "cosine",
    construction_ef: int = 100,
    search_ef: int = 50,
    max_neighbors: int = 16,
) -> Collection:
    try:
        client.delete_collection(name)
    except (ValueError, NotFoundError):
        pass

    metadata = {
        "hnsw:space": space,
        "hnsw:construction_ef": construction_ef,
        "hnsw:search_ef": search_ef,
        "hnsw:M": max_neighbors,
    }

    return client.create_collection(
        name=name,
        embedding_function=embedding_function,
        metadata=metadata,
    )


def index_documents(collection: Collection, records: list[DocumentRecord], batch_size: int = 128) -> None:
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        collection.add(
            ids=[record.id for record in batch],
            documents=[record.text for record in batch],
            metadatas=[
                {
                    "title": record.title,
                    "category": record.category,
                    "source": record.source,
                    "year": record.year,
                    "difficulty": record.difficulty,
                }
                for record in batch
            ],
        )

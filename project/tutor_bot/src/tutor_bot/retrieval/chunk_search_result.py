from pathlib import PurePosixPath
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ChunkSearchResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    chunk_id: UUID
    note_id: UUID
    note_title: str
    section_index: int
    chunk_index: int
    heading_title: str
    heading_path: tuple[str, ...]
    text: str
    theme: str
    relative_path: PurePosixPath
    score: float
    retrieval_method: Literal["vector", "bm25"]

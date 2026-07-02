from pathlib import PurePosixPath
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class IndexedChunk(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    chunk_id: UUID
    note_id: UUID
    section_index: int = Field(ge=0)
    chunk_index: int = Field(ge=0)
    note_title: str = Field(min_length=1)
    heading_title: str
    heading_path: tuple[str, ...]
    text: str = Field(min_length=1)
    theme: str
    importance: int = Field(ge=0, le=10)
    knowledge: int = Field(ge=0, le=10)
    relative_path: PurePosixPath
    source_modified_at: AwareDatetime

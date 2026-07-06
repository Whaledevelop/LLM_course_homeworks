from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class NoteListItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    title: str = Field(min_length=1)
    group: str
    importance: int = Field(ge=0, le=10)
    knowledge: int = Field(ge=0, le=10)
    fullness: int = Field(default=0, ge=0, le=10)
    favorite: bool = False

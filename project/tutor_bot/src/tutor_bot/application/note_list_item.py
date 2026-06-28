from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class NoteListItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    title: str = Field(min_length=1)
    theme: str
    difficulty: str
    importance: int = Field(ge=0, le=10)
    completeness: int = Field(ge=0, le=10)
    mastery: int = Field(ge=0, le=10)

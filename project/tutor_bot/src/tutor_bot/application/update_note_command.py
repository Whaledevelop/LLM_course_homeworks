from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class UpdateNoteCommand(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    note_id: UUID
    title: str = Field(min_length=1)
    group: str
    comment: str
    importance: int = Field(ge=0, le=10)
    knowledge: int = Field(ge=0, le=10)
    markdown_content: str

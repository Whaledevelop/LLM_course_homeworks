from pydantic import BaseModel, ConfigDict, Field


class CreateNoteCommand(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    title: str = Field(min_length=1)
    group: str
    comment: str
    importance: int = Field(ge=0, le=10)
    knowledge: int = Field(ge=0, le=10)
    fullness: int = Field(default=0, ge=0, le=10)
    markdown_content: str

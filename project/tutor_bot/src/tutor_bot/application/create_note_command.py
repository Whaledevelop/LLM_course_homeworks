from pydantic import BaseModel, ConfigDict, Field


class CreateNoteCommand(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    title: str = Field(min_length=1)
    theme: str
    comment: str
    difficulty: str
    importance: int = Field(ge=0, le=10)
    completeness: int = Field(ge=0, le=10)
    mastery: int = Field(ge=0, le=10)
    markdown_content: str = Field(min_length=1)

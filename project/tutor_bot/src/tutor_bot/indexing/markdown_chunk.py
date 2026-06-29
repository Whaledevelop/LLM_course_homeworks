from pydantic import BaseModel, ConfigDict, Field


class MarkdownChunk(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    section_index: int = Field(ge=0)
    chunk_index: int = Field(ge=0)
    heading_level: int = Field(ge=0, le=6)
    heading_title: str
    heading_path: tuple[str, ...]
    content: str = Field(min_length=1)

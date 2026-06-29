from pydantic import BaseModel, ConfigDict, Field


class MarkdownSection(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    heading_level: int = Field(ge=0, le=6)
    heading_title: str
    heading_path: tuple[str, ...]
    content: str = Field(min_length=1)

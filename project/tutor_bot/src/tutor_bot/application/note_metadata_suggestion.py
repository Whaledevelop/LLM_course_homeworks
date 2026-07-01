from pydantic import BaseModel, ConfigDict, Field


class NoteMetadataSuggestion(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    title: str = Field(min_length=1)
    theme: str = Field(min_length=1)
    difficulty: str = Field(min_length=1)
    comment: str = Field(min_length=1)
    key_concepts: tuple[str, ...] = Field(
        min_length=2,
        max_length=8,
    )

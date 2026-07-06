from pydantic import BaseModel, ConfigDict, Field


class NoteMetadataSuggestion(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    title: str = Field(min_length=1)
    group: str = Field(min_length=1)
    questions_for_tests: tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
    )
    importance: int = Field(ge=0, le=10)
    key_concepts: tuple[str, ...] = Field(
        min_length=1,
        max_length=8,
    )

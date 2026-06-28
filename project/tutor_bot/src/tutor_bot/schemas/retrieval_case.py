from typing import Self

from pydantic import BaseModel, Field, model_validator


class RetrievalCase(BaseModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=3)
    expected_note_ids: list[str] = Field(default_factory=list)
    should_find_answer: bool = True

    @model_validator(mode="after")
    def validate_expected_notes(self) -> Self:
        if self.should_find_answer and not self.expected_note_ids:
            raise ValueError("Answerable case must contain expected note ids")

        if not self.should_find_answer and self.expected_note_ids:
            raise ValueError("Unanswerable case cannot contain expected note ids")

        if len(self.expected_note_ids) != len(set(self.expected_note_ids)):
            raise ValueError("Expected note ids must be unique")

        return self
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RecallAnswerReview(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    verdict: Literal[
        "correct",
        "partially_correct",
        "incorrect",
    ]
    covered_points: tuple[str, ...] = Field(max_length=5)
    missing_points: tuple[str, ...] = Field(max_length=5)
    errors: tuple[str, ...] = Field(max_length=5)
    feedback: str = Field(min_length=1)

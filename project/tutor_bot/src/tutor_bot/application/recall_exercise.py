from pydantic import BaseModel, ConfigDict, Field


class RecallExercise(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    question: str = Field(min_length=1)
    key_points: tuple[str, ...] = Field(
        min_length=2,
        max_length=5,
    )
    reference_answer: str = Field(min_length=1)

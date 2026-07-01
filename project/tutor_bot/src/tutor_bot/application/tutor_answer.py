from pydantic import BaseModel, ConfigDict, Field

from tutor_bot.retrieval.context_gate_result import ContextGateResult


class TutorAnswer(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    context: ContextGateResult

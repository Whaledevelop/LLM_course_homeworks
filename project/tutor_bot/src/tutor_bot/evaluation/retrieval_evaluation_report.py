from pydantic import BaseModel, ConfigDict, Field


class RetrievalEvaluationReport(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    case_count: int = Field(gt=0)
    answerable_case_count: int = Field(ge=0)
    recall_k: int = Field(gt=0)
    recall_at_k: float = Field(ge=0, le=1)
    mean_reciprocal_rank: float = Field(ge=0, le=1)
    context_gate_accuracy: float = Field(ge=0, le=1)
    mean_latency_ms: float = Field(ge=0)
    p95_latency_ms: float = Field(ge=0)

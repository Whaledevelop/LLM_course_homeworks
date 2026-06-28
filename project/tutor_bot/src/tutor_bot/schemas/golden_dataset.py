from pydantic import BaseModel, ConfigDict, Field, field_validator

from tutor_bot.schemas.retrieval_case import RetrievalCase


class GoldenDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    retrieval_cases: list[RetrievalCase] = Field(min_length=1)

    @field_validator("retrieval_cases")
    @classmethod
    def validate_unique_case_ids(
        cls,
        retrieval_cases: list[RetrievalCase],
    ) -> list[RetrievalCase]:
        case_ids = [retrieval_case.id for retrieval_case in retrieval_cases]

        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Retrieval case ids must be unique")

        return retrieval_cases

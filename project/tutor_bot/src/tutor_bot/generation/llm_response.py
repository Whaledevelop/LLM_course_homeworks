from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LlmResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    provider: str
    model: str
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    raw_usage: dict[str, Any] | None = None

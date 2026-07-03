from pydantic import BaseModel, ConfigDict


class ObservabilitySinkStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    enabled: bool
    available: bool | None = None
    message: str

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class ChatRoute(StrEnum):
    CAPABILITIES = "capabilities"
    LOCAL = "local"
    GENERAL = "general"
    CREATE_NOTE = "create_note"
    UPDATE_NOTE = "update_note"
    START_RECALL = "start_recall"
    UNAVAILABLE = "unavailable"


class ChatRouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    route: ChatRoute
    note_title: str | None = None
    instruction: str | None = None

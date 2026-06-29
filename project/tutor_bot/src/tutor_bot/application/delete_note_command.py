from uuid import UUID

from pydantic import BaseModel, ConfigDict


class DeleteNoteCommand(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    note_id: UUID

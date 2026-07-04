from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from tutor_bot.application.tutor_answer import TutorAnswer


class CreateNoteDraft(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    title: str = Field(min_length=1)
    markdown_content: str = Field(min_length=1)
    fullness: int = Field(default=7, ge=4, le=10)


class UpdateNoteDraft(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    note_id: UUID
    title: str = Field(min_length=1)
    original_markdown_content: str
    markdown_content: str


class StartRecallDraft(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    note_id: UUID
    title: str = Field(min_length=1)
    requires_title_confirmation: bool = False


class ChatResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    answer: TutorAnswer | None = None
    create_note_draft: CreateNoteDraft | None = None
    update_note_draft: UpdateNoteDraft | None = None
    start_recall_draft: StartRecallDraft | None = None

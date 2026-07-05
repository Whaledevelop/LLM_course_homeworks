from collections.abc import Callable

from tutor_bot.application.active_recall_service import ActiveRecallService
from tutor_bot.application.chat_result import (
    ChatResult,
    CreateNoteDraft,
    StartRecallDraft,
    UpdateNoteDraft,
)
from tutor_bot.application.chat_workflow import ChatWorkflow
from tutor_bot.application.create_note_command import CreateNoteCommand
from tutor_bot.application.note_command_service import NoteCommandService
from tutor_bot.application.note_details import NoteDetails
from tutor_bot.application.note_fullness import estimate_note_fullness
from tutor_bot.application.note_query_service import NoteQueryService
from tutor_bot.application.recall_study_session import RecallStudySession
from tutor_bot.application.tutor_answer_service import TutorAnswerService
from tutor_bot.application.update_note_command import UpdateNoteCommand
from tutor_bot.generation.llm_provider import LlmProvider
from tutor_bot.generation.note_content_generator import NoteContentGenerator


class ChatService:
    def __init__(
        self,
        provider: LlmProvider,
        local_answer_service_factory: Callable[[], TutorAnswerService],
        note_command_service: NoteCommandService | None = None,
        note_content_generator: NoteContentGenerator | None = None,
        note_query_service: NoteQueryService | None = None,
        active_recall_service: ActiveRecallService | None = None,
    ) -> None:
        self._note_command_service = note_command_service
        self._note_query_service = note_query_service
        self._active_recall_service = active_recall_service
        self._workflow = ChatWorkflow(
            provider,
            local_answer_service_factory,
            note_content_generator,
            note_query_service,
        )

    def answer(self, question: str) -> ChatResult:
        return self._workflow.invoke(question.strip())

    def create_note(self, draft: CreateNoteDraft) -> NoteDetails:
        if self._note_command_service is None:
            raise RuntimeError("Note command service is not configured")

        return self._note_command_service.create_note(
            CreateNoteCommand(
                title=draft.title,
                group="",
                comment="Создано через Chat",
                questions_for_tests=(),
                importance=5,
                knowledge=0,
                fullness=estimate_note_fullness(draft.markdown_content),
                markdown_content=draft.markdown_content,
            )
        )

    def update_note(self, draft: UpdateNoteDraft) -> NoteDetails:
        if self._note_query_service is None or self._note_command_service is None:
            raise RuntimeError("Note services are not configured")

        note = self._note_query_service.get_note(draft.note_id)

        if note.markdown_content != draft.original_markdown_content:
            raise RuntimeError("Заметка изменилась после формирования предпросмотра")

        return self._note_command_service.update_note(
            UpdateNoteCommand(
                note_id=note.id,
                title=note.title,
                group=note.group,
                comment=note.comment,
                questions_for_tests=note.questions_for_tests,
                importance=note.importance,
                knowledge=note.knowledge,
                fullness=estimate_note_fullness(draft.markdown_content),
                markdown_content=draft.markdown_content,
            )
        )

    def start_recall(self, draft: StartRecallDraft) -> RecallStudySession:
        if self._active_recall_service is None:
            raise RuntimeError("Active Recall service is not configured")

        return self._active_recall_service.create_note_study_session(draft.note_id)

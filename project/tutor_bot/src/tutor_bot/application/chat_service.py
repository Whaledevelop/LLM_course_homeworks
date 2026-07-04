from collections.abc import Callable
import re
from typing import TypedDict

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from tutor_bot.application.active_recall_service import ActiveRecallService
from tutor_bot.application.chat_result import (
    ChatResult,
    CreateNoteDraft,
    StartRecallDraft,
    UpdateNoteDraft,
)
from tutor_bot.application.chat_route import ChatRoute, ChatRouteDecision
from tutor_bot.application.create_note_command import CreateNoteCommand
from tutor_bot.application.note_command_service import NoteCommandService
from tutor_bot.application.note_details import NoteDetails
from tutor_bot.application.note_fullness import estimate_note_fullness
from tutor_bot.application.note_query_service import NoteQueryService
from tutor_bot.application.recall_study_session import RecallStudySession
from tutor_bot.application.tutor_answer import TutorAnswer
from tutor_bot.application.tutor_answer_service import TutorAnswerService
from tutor_bot.application.update_note_command import UpdateNoteCommand
from tutor_bot.generation.llm_provider import LlmProvider
from tutor_bot.generation.note_content_generator import NoteContentGenerator
from tutor_bot.retrieval.context_gate_result import ContextGateResult


_UNAVAILABLE_ANSWER = "Ответ недоступен"
_ROUTER_PROMPT = """Определи, как обработать запрос пользователя в учебном приложении.
local: пользователь явно спрашивает о своих заметках, локальной базе знаний, изученных материалах или просит найти информацию именно в них.
general: на запрос можно ответить с помощью общих знаний LLM без локальных заметок.
create_note: пользователь просит создать или добавить заметку в активную базу. Извлеки название заметки в note_title. Не придумывай название, если пользователь его не указал.
update_note: пользователь просит дополнить или изменить существующую заметку. Извлеки точное название в note_title и пожелание к изменению в instruction.
start_recall: пользователь просит запустить тест или Active Recall по существующей заметке. Извлеки точное название в note_title.
unavailable: запрос не является вопросом к локальным знаниям или LLM либо требует выполнить недоступное действие.
{format_instructions}"""
_GENERAL_PROMPT = """Ты помощник в учебном приложении.
Ответь на вопрос с помощью своих общих знаний.
Не утверждай, что использовал локальные заметки или внешний поиск.
Если достоверного ответа нет, скажи об этом прямо.
Отвечай на языке пользователя."""
_CREATE_NOTE_PATTERN = re.compile(
    r"(?:добав\w*|созда\w*).*?заметк\w*.*?(?:с названием|названи\w*)\s*[\"«](?P<title>.+?)[\"»]",
    re.IGNORECASE,
)
_QUOTED_TITLE_PATTERN = re.compile(r"[\"«](?P<title>.+?)[\"»]")


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
        self._provider = provider
        self._local_answer_service_factory = local_answer_service_factory
        self._note_command_service = note_command_service
        self._note_content_generator = note_content_generator
        self._note_query_service = note_query_service
        self._active_recall_service = active_recall_service
        self._router_parser = PydanticOutputParser(pydantic_object=ChatRouteDecision)
        self._graph = self._create_graph()

    def answer(self, question: str) -> ChatResult:
        normalized_question = question.strip()
        state = self._graph.invoke({"question": normalized_question})

        return ChatResult(
            answer=state.get("answer"),
            create_note_draft=state.get("create_note_draft"),
            update_note_draft=state.get("update_note_draft"),
            start_recall_draft=state.get("start_recall_draft"),
        )

    def create_note(self, draft: CreateNoteDraft) -> NoteDetails:
        if self._note_command_service is None:
            raise RuntimeError("Note command service is not configured")

        return self._note_command_service.create_note(
            CreateNoteCommand(
                title=draft.title,
                group="",
                comment="Создано через Chat",
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

    def _create_graph(self):
        graph = StateGraph(_ChatState)
        graph.add_node("supervisor", self._run_supervisor)
        graph.add_node("local", self._run_local_agent)
        graph.add_node("general", self._run_general_agent)
        graph.add_node("create_note", self._run_create_note_agent)
        graph.add_node("update_note", self._run_update_note_agent)
        graph.add_node("start_recall", self._run_start_recall_agent)
        graph.add_node("unavailable", self._run_unavailable)
        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._select_node,
            {
                ChatRoute.LOCAL: "local",
                ChatRoute.GENERAL: "general",
                ChatRoute.CREATE_NOTE: "create_note",
                ChatRoute.UPDATE_NOTE: "update_note",
                ChatRoute.START_RECALL: "start_recall",
                ChatRoute.UNAVAILABLE: "unavailable",
            },
        )
        graph.add_edge("local", END)
        graph.add_edge("general", END)
        graph.add_edge("create_note", END)
        graph.add_edge("update_note", END)
        graph.add_edge("start_recall", END)
        graph.add_edge("unavailable", END)

        return graph.compile()

    def _run_supervisor(self, state: "_ChatState") -> "_ChatState":
        return {"decision": self._select_route(state["question"])}

    def _select_node(self, state: "_ChatState") -> ChatRoute:
        return state["decision"].route

    def _run_local_agent(self, state: "_ChatState") -> "_ChatState":
        answer = self._local_answer_service_factory().answer(state["question"])

        return {"answer": answer}

    def _run_general_agent(self, state: "_ChatState") -> "_ChatState":
        context = ContextGateResult(
            selected_results=(),
            minimum_reranker_score=0.0,
        )
        answer = TutorAnswer(
            question=state["question"],
            answer=self._answer_from_general_knowledge(state["question"]),
            context=context,
        )

        return {"answer": answer}

    def _run_create_note_agent(self, state: "_ChatState") -> "_ChatState":
        decision = state["decision"]

        if self._note_content_generator is None or not decision.note_title:
            return {"answer": self._create_unavailable_answer(state["question"])}

        markdown_content = self._note_content_generator.generate(
            decision.note_title,
            fullness=7,
        )

        return {
            "create_note_draft": CreateNoteDraft(
                title=decision.note_title,
                markdown_content=markdown_content,
            )
        }

    def _run_update_note_agent(self, state: "_ChatState") -> "_ChatState":
        decision = state["decision"]

        if self._note_content_generator is None or not decision.note_title:
            return {"answer": self._create_unavailable_answer(state["question"])}

        note = self._find_note(decision.note_title)

        if note is None:
            return {"answer": self._create_note_not_found_answer(state["question"])}

        generation_title = note.title

        if decision.instruction:
            generation_title = f"{note.title}. Требуемое изменение: {decision.instruction}"

        markdown_content = self._note_content_generator.generate(
            generation_title,
            existing_content=note.markdown_content,
            fullness=max(7, note.fullness),
        )

        return {
            "update_note_draft": UpdateNoteDraft(
                note_id=note.id,
                title=note.title,
                original_markdown_content=note.markdown_content,
                markdown_content=markdown_content,
            )
        }

    def _run_start_recall_agent(self, state: "_ChatState") -> "_ChatState":
        decision = state["decision"]

        if not decision.note_title:
            return {"answer": self._create_unavailable_answer(state["question"])}

        note = self._find_note(decision.note_title)

        if note is None:
            return {"answer": self._create_note_not_found_answer(state["question"])}

        return {
            "start_recall_draft": StartRecallDraft(
                note_id=note.id,
                title=note.title,
            )
        }

    def _run_unavailable(self, state: "_ChatState") -> "_ChatState":
        return {"answer": self._create_unavailable_answer(state["question"])}

    def _create_unavailable_answer(self, question: str) -> TutorAnswer:
        context = ContextGateResult(
            selected_results=(),
            minimum_reranker_score=0.0,
        )

        return TutorAnswer(
            question=question,
            answer=_UNAVAILABLE_ANSWER,
            context=context,
        )

    def _create_note_not_found_answer(self, question: str) -> TutorAnswer:
        context = ContextGateResult(
            selected_results=(),
            minimum_reranker_score=0.0,
        )

        return TutorAnswer(
            question=question,
            answer="Заметка с таким точным названием не найдена",
            context=context,
        )

    def _find_note(self, title: str) -> NoteDetails | None:
        if self._note_query_service is None:
            return None

        normalized_title = title.casefold().strip()
        matched_notes = [
            note
            for note in self._note_query_service.list_notes()
            if note.title.casefold().strip() == normalized_title
        ]

        if len(matched_notes) != 1:
            return None

        return self._note_query_service.get_note(matched_notes[0].id)

    def _select_route(self, question: str) -> ChatRouteDecision:
        quoted_title_match = _QUOTED_TITLE_PATTERN.search(question)
        quoted_title = (
            quoted_title_match.group("title").strip()
            if quoted_title_match is not None
            else None
        )
        normalized_question = question.casefold()

        if quoted_title and ("active recall" in normalized_question or "тест" in normalized_question):
            return ChatRouteDecision(
                route=ChatRoute.START_RECALL,
                note_title=quoted_title,
            )

        if quoted_title and any(
            marker in normalized_question
            for marker in ("дополни", "обнови", "измени", "добавь в заметку")
        ):
            return ChatRouteDecision(
                route=ChatRoute.UPDATE_NOTE,
                note_title=quoted_title,
                instruction=question,
            )

        create_note_match = _CREATE_NOTE_PATTERN.search(question)

        if create_note_match is not None:
            return ChatRouteDecision(
                route=ChatRoute.CREATE_NOTE,
                note_title=create_note_match.group("title").strip(),
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _ROUTER_PROMPT),
                ("human", "{question}"),
            ]
        )
        messages = prompt.format_messages(
            question=question,
            format_instructions=self._router_parser.get_format_instructions(),
        )
        response = self._provider.generate(
            messages=[
                {
                    "role": "system" if message.type == "system" else "user",
                    "content": str(message.content),
                }
                for message in messages
            ],
            temperature=0.0,
            max_tokens=200,
            json_schema=ChatRouteDecision.model_json_schema(),
        )

        return self._router_parser.parse(response.text)

    def _answer_from_general_knowledge(self, question: str) -> str:
        response = self._provider.generate(
            messages=[
                {"role": "system", "content": _GENERAL_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.2,
            max_tokens=1200,
        )

        return response.text.strip()


class _ChatState(TypedDict, total=False):
    question: str
    decision: ChatRouteDecision
    answer: TutorAnswer
    create_note_draft: CreateNoteDraft
    update_note_draft: UpdateNoteDraft
    start_recall_draft: StartRecallDraft

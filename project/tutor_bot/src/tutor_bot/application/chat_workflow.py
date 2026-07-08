from collections.abc import Callable
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from tutor_bot.application.chat_result import (
    ChatResult,
    CreateNoteDraft,
    StartRecallDraft,
    UpdateNoteDraft,
)
from tutor_bot.application.chat_route import ChatRoute, ChatRouteDecision
from tutor_bot.application.chat_supervisor import ChatSupervisor
from tutor_bot.application.chat_source_query import (
    detect_explicit_chat_source,
    strip_chat_source_instruction,
)
from tutor_bot.application.llm_note_title_matcher import LlmNoteTitleMatcher
from tutor_bot.application.note_details import NoteDetails
from tutor_bot.application.note_query_service import NoteQueryService
from tutor_bot.application.tutor_answer import TutorAnswer
from tutor_bot.application.tutor_answer_service import TutorAnswerService
from tutor_bot.generation.llm_provider import LlmProvider
from tutor_bot.generation.note_content_generator import NoteContentGenerator
from tutor_bot.retrieval.context_gate_result import ContextGateResult


_UNAVAILABLE_ANSWER = "Ответ недоступен"
_CAPABILITIES_RESPONSE = """Поддерживаемые запросы:

1. Ответить по локальным заметкам с помощью RAG. Например: «Используя только локальные заметки, расскажи про мультиплеер в Unity».
2. Ответить из общих знаний выбранной LLM. Например: «Используя знания LLM, объясни принципы SOLID».
3. Создать заметку с предварительным просмотром. Например: «Создай заметку с названием „Dependency Injection“».
4. Дополнить существующую заметку с подтверждением изменений. Например: «Дополни заметку „SOLID“ примерами на C#».
5. Запустить Test Notes по конкретной заметке. Например: «Запусти тест по заметке „SOLID“».

Вопрос или команду можно ввести текстом либо надиктовать через speech-to-text."""
_GENERAL_PROMPT = """Ты помощник в учебном приложении.
Ответь на вопрос с помощью своих общих знаний.
Не утверждай, что использовал локальные заметки или внешний поиск.
Если достоверного ответа нет, скажи об этом прямо.
Отвечай на языке пользователя."""


class ChatWorkflow:
    def __init__(
        self,
        provider: LlmProvider,
        local_answer_service_factory: Callable[[], TutorAnswerService],
        note_content_generator: NoteContentGenerator | None,
        note_query_service: NoteQueryService | None,
    ) -> None:
        self._provider = provider
        self._local_answer_service_factory = local_answer_service_factory
        self._note_content_generator = note_content_generator
        self._note_query_service = note_query_service
        self._supervisor = ChatSupervisor(provider)
        self._note_title_matcher = LlmNoteTitleMatcher(provider)
        self._graph = self._create_graph()

    def invoke(self, question: str) -> ChatResult:
        state = self._graph.invoke({"question": question})

        return ChatResult(
            answer=state.get("answer"),
            create_note_draft=state.get("create_note_draft"),
            update_note_draft=state.get("update_note_draft"),
            start_recall_draft=state.get("start_recall_draft"),
        )

    def _create_graph(self):
        graph = StateGraph(_ChatState)
        graph.add_node("supervisor", self._run_supervisor)
        graph.add_node("capabilities", self._run_capabilities_agent)
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
            {route: route.value for route in ChatRoute},
        )

        for route in ChatRoute:
            graph.add_edge(route.value, END)

        return graph.compile()

    def _run_supervisor(self, state: "_ChatState") -> "_ChatState":
        return {"decision": self._supervisor.select_route(state["question"])}

    def _select_node(self, state: "_ChatState") -> ChatRoute:
        return state["decision"].route

    def _run_capabilities_agent(self, state: "_ChatState") -> "_ChatState":
        return {
            "answer": self._create_answer(
                state["question"],
                _CAPABILITIES_RESPONSE,
            )
        }

    def _run_local_agent(self, state: "_ChatState") -> "_ChatState":
        question = strip_chat_source_instruction(state["question"])
        answer = self._local_answer_service_factory().answer(question)

        if (
            detect_explicit_chat_source(state["question"]) is None
            and not answer.context.has_sufficient_context
        ):
            return {"answer": self._answer_from_general_llm(question)}

        return {"answer": answer}

    def _run_general_agent(self, state: "_ChatState") -> "_ChatState":
        question = strip_chat_source_instruction(state["question"])

        return {"answer": self._answer_from_general_llm(question)}

    def _answer_from_general_llm(self, question: str) -> TutorAnswer:
        response = self._provider.generate(
            messages=[
                {"role": "system", "content": _GENERAL_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.2,
            max_tokens=1200,
        )

        return self._create_answer(question, response.text.strip())

    def _run_create_note_agent(self, state: "_ChatState") -> "_ChatState":
        decision = state["decision"]

        if self._note_content_generator is None or not decision.note_title:
            return self._run_unavailable(state)

        return {
            "create_note_draft": CreateNoteDraft(
                title=decision.note_title,
                markdown_content=self._note_content_generator.generate(
                    decision.note_title,
                    fullness=7,
                ),
            )
        }

    def _run_update_note_agent(self, state: "_ChatState") -> "_ChatState":
        decision = state["decision"]

        if self._note_content_generator is None or not decision.note_title:
            return self._run_unavailable(state)

        note = self._find_note(decision.note_title)

        if note is None:
            return {"answer": self._create_note_not_found_answer(state["question"])}

        generation_title = note.title

        if decision.instruction:
            generation_title = f"{note.title}. Требуемое изменение: {decision.instruction}"

        return {
            "update_note_draft": UpdateNoteDraft(
                note_id=note.id,
                title=note.title,
                original_markdown_content=note.markdown_content,
                markdown_content=self._note_content_generator.generate(
                    generation_title,
                    existing_content=note.markdown_content,
                    fullness=max(7, note.fullness),
                ),
            )
        }

    def _run_start_recall_agent(self, state: "_ChatState") -> "_ChatState":
        decision = state["decision"]
        note, requires_title_confirmation = self._find_note_match(
            decision.note_title or ""
        )

        if note is None:
            return {"answer": self._create_note_not_found_answer(state["question"])}

        return {
            "start_recall_draft": StartRecallDraft(
                note_id=note.id,
                title=note.title,
                requires_title_confirmation=requires_title_confirmation,
            )
        }

    def _run_unavailable(self, state: "_ChatState") -> "_ChatState":
        return {"answer": self._create_answer(state["question"], _UNAVAILABLE_ANSWER)}

    def _create_note_not_found_answer(self, question: str) -> TutorAnswer:
        return self._create_answer(question, "Заметка с таким точным названием не найдена")

    def _create_answer(self, question: str, answer: str) -> TutorAnswer:
        return TutorAnswer(
            question=question,
            answer=answer,
            context=ContextGateResult(selected_results=(), minimum_reranker_score=0.0),
        )

    def _find_note(self, title: str) -> NoteDetails | None:
        note, _ = self._find_note_match(title)

        return note

    def _find_note_match(self, title: str) -> tuple[NoteDetails | None, bool]:
        if self._note_query_service is None:
            return None, False

        notes = self._note_query_service.list_notes()
        normalized_title = title.casefold().strip()
        exact_matches = [
            note
            for note in notes
            if note.title.casefold().strip() == normalized_title
        ]

        if len(exact_matches) == 1:
            return self._note_query_service.get_note(exact_matches[0].id), False

        matched_title = self._note_title_matcher.match(
            title,
            [note.title for note in notes],
        )
        matched_notes = [note for note in notes if note.title == matched_title]

        if len(matched_notes) != 1:
            return None, False

        return self._note_query_service.get_note(matched_notes[0].id), True


class _ChatState(TypedDict, total=False):
    question: str
    decision: ChatRouteDecision
    answer: TutorAnswer
    create_note_draft: CreateNoteDraft
    update_note_draft: UpdateNoteDraft
    start_recall_draft: StartRecallDraft

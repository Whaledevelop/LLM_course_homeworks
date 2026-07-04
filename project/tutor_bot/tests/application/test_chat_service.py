from types import SimpleNamespace
from uuid import uuid4

from tutor_bot.application.chat_service import ChatService
from tutor_bot.application.note_details import NoteDetails
from tutor_bot.application.tutor_answer import TutorAnswer
from tutor_bot.generation.llm_response import LlmResponse
from tutor_bot.retrieval.context_gate_result import ContextGateResult


class _FakeProvider:
    provider_name = "fake"
    model_name = "fake-model"

    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_schema: dict[str, object] | None = None,
    ) -> LlmResponse:
        return LlmResponse(
            text=next(self._responses),
            provider=self.provider_name,
            model=self.model_name,
        )


class _LocalAnswerService:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def answer(self, question: str):
        self.questions.append(question)

        return TutorAnswer(
            question=question,
            answer="local-answer",
            context=ContextGateResult(
                selected_results=(),
                minimum_reranker_score=0.0,
            ),
        )


class _NoteContentGenerator:
    def generate(
        self,
        title: str,
        existing_content: str = "",
        fullness: int = 7,
    ) -> str:
        return f"## Содержание\n\nМатериал о {title}"


class _NoteCommandService:
    def __init__(self) -> None:
        self.commands = []

    def create_note(self, command):
        self.commands.append(command)

        return SimpleNamespace(title=command.title, id="note-id")

    def update_note(self, command):
        self.commands.append(command)

        return SimpleNamespace(title=command.title, id=command.note_id)


class _NoteQueryService:
    def __init__(self, note: NoteDetails, *other_notes: NoteDetails) -> None:
        self._notes = [note, *other_notes]

    def list_notes(self):
        return self._notes

    def get_note(self, note_id):
        return next(note for note in self._notes if note.id == note_id)


class _ActiveRecallService:
    def __init__(self) -> None:
        self.note_ids = []

    def create_note_study_session(self, note_id):
        self.note_ids.append(note_id)

        return "study-session"


def test_local_route_loads_local_service() -> None:
    local_service = _LocalAnswerService()
    service = ChatService(
        _FakeProvider(['{"route":"local"}']),
        lambda: local_service,
    )

    result = service.answer("Что сказано в моих заметках?")

    assert result.answer.answer == "local-answer"
    assert local_service.questions == ["Что сказано в моих заметках?"]


def test_general_route_answers_without_loading_local_service() -> None:
    local_service_loaded = False

    def create_local_service():
        nonlocal local_service_loaded
        local_service_loaded = True

        return _LocalAnswerService()

    service = ChatService(
        _FakeProvider(['{"route":"general"}', "Общий ответ"]),
        create_local_service,
    )

    result = service.answer("Что такое Python?")

    assert result.answer.answer == "Общий ответ"
    assert not result.answer.context.has_sufficient_context
    assert local_service_loaded is False


def test_unavailable_route_returns_fixed_answer() -> None:
    service = ChatService(
        _FakeProvider(['{"route":"unavailable"}']),
        lambda: _LocalAnswerService(),
    )

    result = service.answer("Выполни недоступное действие")

    assert result.answer.answer == "Ответ недоступен"


def test_create_note_route_returns_draft_without_writing() -> None:
    note_command_service = _NoteCommandService()
    service = ChatService(
        _FakeProvider(
            ['{"route":"create_note","note_title":"LLM в Unity разработке"}']
        ),
        lambda: _LocalAnswerService(),
        note_command_service=note_command_service,
        note_content_generator=_NoteContentGenerator(),
    )

    result = service.answer(
        'Добавь заметку с названием "LLM в Unity разработке" и сгенерируй контент'
    )

    assert result.answer is None
    assert result.create_note_draft.title == "LLM в Unity разработке"
    assert "Материал о LLM в Unity разработке" in result.create_note_draft.markdown_content
    assert note_command_service.commands == []

    created_note = service.create_note(result.create_note_draft)

    assert created_note.title == "LLM в Unity разработке"
    assert note_command_service.commands[0].comment == "Создано через Chat"


def test_update_note_route_returns_draft_and_updates_after_confirmation() -> None:
    note = _create_note("LLM в Unity разработке")
    note_command_service = _NoteCommandService()
    service = ChatService(
        _FakeProvider([]),
        lambda: _LocalAnswerService(),
        note_command_service=note_command_service,
        note_content_generator=_NoteContentGenerator(),
        note_query_service=_NoteQueryService(note),
    )

    result = service.answer(
        'Дополни заметку "LLM в Unity разработке" примерами интеграции'
    )

    assert result.update_note_draft.title == note.title
    assert note_command_service.commands == []

    service.update_note(result.update_note_draft)

    assert note_command_service.commands[0].note_id == note.id
    assert "Материал о" in note_command_service.commands[0].markdown_content


def test_start_recall_route_prepares_selected_note() -> None:
    note = _create_note("LLM в Unity разработке")
    active_recall_service = _ActiveRecallService()
    service = ChatService(
        _FakeProvider([]),
        lambda: _LocalAnswerService(),
        note_query_service=_NoteQueryService(note),
        active_recall_service=active_recall_service,
    )

    result = service.answer('Запусти Active Recall по заметке "LLM в Unity разработке"')

    assert result.start_recall_draft.note_id == note.id
    assert result.start_recall_draft.requires_title_confirmation is False
    assert active_recall_service.note_ids == []

    study_session = service.start_recall(result.start_recall_draft)

    assert study_session == "study-session"
    assert active_recall_service.note_ids == [note.id]


def test_start_recall_route_matches_partial_note_title_with_typo_using_llm() -> None:
    expected_note = _create_note("Addressables, AssetBundles, Resources")
    other_note = _create_note("Unity UI Toolkit")
    service = ChatService(
        _FakeProvider(
            [
                '{"route":"start_recall","note_title":"Adrressables"}',
                '{"matched_title":"Addressables, AssetBundles, Resources"}',
            ]
        ),
        lambda: _LocalAnswerService(),
        note_query_service=_NoteQueryService(expected_note, other_note),
        active_recall_service=_ActiveRecallService(),
    )

    result = service.answer("Начни Active Recall по заметке Adrressables")

    assert result.start_recall_draft.note_id == expected_note.id
    assert result.start_recall_draft.title == expected_note.title
    assert result.start_recall_draft.requires_title_confirmation is True


def test_start_recall_route_rejects_title_not_present_in_available_notes() -> None:
    note = _create_note("Addressables, AssetBundles, Resources")
    service = ChatService(
        _FakeProvider(
            [
                '{"route":"start_recall","note_title":"несуществующая"}',
                '{"matched_title":"Придуманная заметка"}',
            ]
        ),
        lambda: _LocalAnswerService(),
        note_query_service=_NoteQueryService(note),
        active_recall_service=_ActiveRecallService(),
    )

    result = service.answer("Начни Active Recall по несуществующей заметке")

    assert result.start_recall_draft is None
    assert result.answer.answer == "Заметка с таким точным названием не найдена"


def _create_note(title: str) -> NoteDetails:
    return NoteDetails(
        id=uuid4(),
        title=title,
        group="AI",
        importance=5,
        knowledge=1,
        fullness=7,
        comment="",
        markdown_content="Исходное содержание",
    )

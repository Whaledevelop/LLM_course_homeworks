from unittest.mock import Mock
from uuid import uuid4

from tutor_bot.application.chat_result import ChatResult
from tutor_bot.application.note_details import NoteDetails
from tutor_bot.application.tutor_answer import TutorAnswer
from tutor_bot.retrieval.context_gate_result import ContextGateResult
from tutor_bot.ui.views import questions_page


def test_save_current_answer_request_is_detected() -> None:
    assert questions_page._is_save_current_answer_request("Сохрани это в заметку")


def test_note_draft_title_is_derived_from_previous_question() -> None:
    draft = questions_page._create_note_draft_from_answer(
        _create_answer("РАсскажи про NavMesh", "NavMesh content"),
        None,
    )

    assert draft.title == "NavMesh"
    assert draft.markdown_content == "NavMesh content"


def test_note_draft_title_can_be_overridden_from_save_request() -> None:
    draft = questions_page._create_note_draft_from_answer(
        _create_answer("Расскажи про NavMesh", "NavMesh content"),
        questions_page._extract_note_title('Сохрани это в заметку "Unity NavMesh"'),
    )

    assert draft.title == "Unity NavMesh"


def test_save_current_answer_creates_note_from_previous_answer(monkeypatch) -> None:
    service = Mock()
    service.create_note.return_value = NoteDetails(
        id=uuid4(),
        title="NavMesh",
        group="",
        comment="",
        importance=5,
        knowledge=0,
        fullness=7,
        markdown_content="NavMesh content",
    )
    session_state = {
        questions_page._ANSWER_KEY: ChatResult(
            answer=_create_answer("Расскажи про NavMesh", "NavMesh content"),
        )
    }

    monkeypatch.setattr(questions_page.st, "session_state", session_state)

    questions_page._save_current_answer_as_note(
        lambda: service,
        "Сохрани это в заметку",
    )

    draft = service.create_note.call_args.args[0]

    assert draft.title == "NavMesh"
    assert draft.markdown_content == "NavMesh content"
    assert session_state[questions_page._ANSWER_KEY].answer.answer.startswith("Заметка")


def _create_answer(question: str, answer: str) -> TutorAnswer:
    return TutorAnswer(
        question=question,
        answer=answer,
        context=ContextGateResult(
            selected_results=(),
            minimum_reranker_score=0.0,
        ),
    )

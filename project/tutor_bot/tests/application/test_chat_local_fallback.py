from tutor_bot.application.chat_workflow import ChatWorkflow
from tutor_bot.application.tutor_answer import TutorAnswer
from tutor_bot.retrieval.context_gate_result import ContextGateResult


class _ProviderResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _Provider:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def generate(self, **kwargs):
        self.questions.append(kwargs["messages"][-1]["content"])

        return _ProviderResponse("Ответ из LLM")


class _FailingProvider:
    def generate(self, **kwargs):
        raise AssertionError("Explicit local source must not fall back to general LLM")


class _LocalAnswerService:
    def __init__(self, answer: TutorAnswer) -> None:
        self.answer_result = answer

    def answer(self, question: str) -> TutorAnswer:
        return self.answer_result


def test_local_route_falls_back_to_general_answer_when_default_context_is_empty() -> None:
    provider = _Provider()
    workflow = ChatWorkflow(
        provider,
        lambda: _LocalAnswerService(_create_local_answer("Что такое SOLID?")),
        None,
        None,
    )

    result = workflow.invoke("Что такое SOLID?")

    assert result.answer is not None
    assert result.answer.answer == "Ответ из LLM"
    assert provider.questions == ["Что такое SOLID?"]


def test_explicit_local_route_does_not_fall_back_to_general_answer() -> None:
    workflow = ChatWorkflow(
        _FailingProvider(),
        lambda: _LocalAnswerService(_create_local_answer("Что такое SOLID")),
        None,
        None,
    )

    result = workflow.invoke("Что такое SOLID согласно локальным заметкам?")

    assert result.answer is not None
    assert result.answer.answer == "В заметках недостаточно информации"


def _create_local_answer(question: str) -> TutorAnswer:
    return TutorAnswer(
        question=question,
        answer="В заметках недостаточно информации",
        context=ContextGateResult(selected_results=(), minimum_reranker_score=0.0),
    )

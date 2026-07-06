from tutor_bot.application.chat_route import ChatRoute
from tutor_bot.application.chat_supervisor import ChatSupervisor
from tutor_bot.application.chat_workflow import ChatWorkflow


class _FailingProvider:
    def generate(self, **kwargs):
        raise AssertionError("Capabilities request must not call an LLM")


def test_capabilities_request_is_routed_without_llm() -> None:
    supervisor = ChatSupervisor(_FailingProvider())

    decision = supervisor.select_route("Напиши список своего функционала")

    assert decision.route == ChatRoute.CAPABILITIES


def test_capabilities_response_lists_supported_chat_requests() -> None:
    workflow = ChatWorkflow(
        _FailingProvider(),
        lambda: (_ for _ in ()).throw(AssertionError("RAG must not be initialized")),
        None,
        None,
    )

    result = workflow.invoke("Какие запросы ты поддерживаешь?")

    assert result.answer is not None
    assert "локальным заметкам" in result.answer.answer
    assert "общих знаний" in result.answer.answer
    assert "Создать заметку" in result.answer.answer
    assert "Дополнить существующую заметку" in result.answer.answer
    assert "Запустить Test Notes" in result.answer.answer

from tutor_bot.application.chat_route import ChatRoute
from tutor_bot.application.chat_source_query import (
    detect_explicit_chat_source,
    strip_chat_source_instruction,
)
from tutor_bot.application.chat_supervisor import ChatSupervisor


class _FailingProvider:
    def generate(self, **kwargs):
        raise AssertionError("Explicit source selection must not call the LLM router")


def test_local_notes_instruction_selects_local_source_and_keeps_topic() -> None:
    question = "Используя только локальные заметки, расскажи про мультиплеер в Unity"

    assert detect_explicit_chat_source(question) == "local"
    assert strip_chat_source_instruction(question) == "расскажи про мультиплеер в Unity"


def test_local_notes_instruction_routes_without_llm_classification() -> None:
    question = "Используя только локальные заметки, расскажи про мультиплеер в Unity"

    decision = ChatSupervisor(_FailingProvider()).select_route(question)

    assert decision.route == ChatRoute.LOCAL


def test_llm_knowledge_instruction_selects_general_source_and_keeps_topic() -> None:
    question = "Используя знания LLM, расскажи про мультиплеер в Unity"

    assert detect_explicit_chat_source(question) == "general"
    assert strip_chat_source_instruction(question) == "расскажи про мультиплеер в Unity"


def test_trailing_local_notes_instruction_is_removed_from_search_query() -> None:
    question = "Что такое SOLID согласно локальным заметкам?"

    assert detect_explicit_chat_source(question) == "local"
    assert strip_chat_source_instruction(question) == "Что такое SOLID"


def test_without_local_notes_takes_precedence_over_local_marker() -> None:
    question = "Без локальных заметок, ответь из общих знаний"

    assert detect_explicit_chat_source(question) == "general"

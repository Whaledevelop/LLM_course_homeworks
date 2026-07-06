import json

from tutor_bot.generation.llm_response import LlmResponse
from tutor_bot.generation.llm_note_metadata_suggester import (
    LlmNoteMetadataSuggester,
)


class _Provider:
    provider_name = "test"
    model_name = "test-model"

    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_schema: dict[str, object] | None = None,
    ) -> LlmResponse:
        self.messages = messages
        text = json.dumps(
            {
                "title": "Индексы базы данных",
                "group": "Базы данных",
                "questions_for_tests": [
                    "Что такое индекс?",
                    "Для чего нужен индекс?",
                    "Как индекс ускоряет поиск?",
                    "Когда индекс замедляет запись?",
                    "Как выбрать поля для индекса?",
                ],
                "importance": 8,
                "key_concepts": ["индекс", "поиск"],
            },
            ensure_ascii=False,
        )

        return LlmResponse(
            text=text,
            provider=self.provider_name,
            model=self.model_name,
        )


def test_suggest_uses_existing_groups_and_returns_extended_metadata() -> None:
    provider = _Provider()
    suggester = LlmNoteMetadataSuggester(provider)

    suggestion = suggester.suggest(
        "# Индексы\nИндекс ускоряет поиск.",
        ("Python", "Базы данных"),
    )

    assert suggestion.group == "Базы данных"
    assert suggestion.importance == 8
    assert len(suggestion.questions_for_tests) == 5
    assert "Базы данных" in provider.messages[1]["content"]
    assert "comment" not in suggestion.model_dump()

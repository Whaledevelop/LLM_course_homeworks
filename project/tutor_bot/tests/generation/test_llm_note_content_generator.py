from tutor_bot.generation.llm_note_content_generator import LlmNoteContentGenerator
from tutor_bot.generation.llm_response import LlmResponse


class _FakeProvider:
    provider_name = "fake"
    model_name = "fake-model"

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.messages: list[dict[str, str]] = []
        self.max_tokens = 0

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_schema: dict[str, object] | None = None,
    ) -> LlmResponse:
        self.messages = messages
        self.max_tokens = max_tokens

        return LlmResponse(
            text=self._response_text,
            provider=self.provider_name,
            model=self.model_name,
        )


def test_generate_content_from_title() -> None:
    provider = _FakeProvider("## Основы\n\nНовый текст")
    generator = LlmNoteContentGenerator(provider)

    result = generator.generate("Тестирование Python")

    assert result == "## Основы\n\nНовый текст"
    assert "Тема заметки: Тестирование Python" in provider.messages[1]["content"]


def test_expand_content_preserves_existing_markdown() -> None:
    provider = _FakeProvider("## Дополнение\n\nНовый текст")
    generator = LlmNoteContentGenerator(provider)
    existing_content = "## Исходный раздел\n\nСтарый текст"

    result = generator.generate("Тестирование Python", existing_content)

    assert result == ("## Исходный раздел\n\nСтарый текст\n\n## Дополнение\n\nНовый текст")
    assert existing_content in provider.messages[1]["content"]


def test_fullness_controls_prompt_and_token_limit() -> None:
    provider = _FakeProvider("Подробное содержание")
    generator = LlmNoteContentGenerator(provider)

    generator.generate("Архитектура", fullness=10)

    assert "Целевая заполненность: 10 из 10" in provider.messages[1]["content"]
    assert provider.max_tokens == 7000

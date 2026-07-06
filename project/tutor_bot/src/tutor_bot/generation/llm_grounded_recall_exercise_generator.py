from pydantic import ValidationError

from tutor_bot.application.recall_exercise import RecallExercise
from tutor_bot.generation.llm_provider import LlmProvider


_SYSTEM_PROMPT = """Ты создаёшь упражнение Active Recall по одной учебной заметке.
Используй только информацию из переданной заметки и не добавляй внешние факты.
Если передан готовый вопрос, используй его без изменений. Иначе сформулируй один сфокусированный открытый вопрос по одной основной теме заметки.
Не объединяй в одном вопросе больше двух тесно связанных понятий.
Вопрос не должен предполагать ответ да или нет.
Выдели от двух до пяти коротких обязательных неповторяющихся тезисов правильного ответа.
Каждая требуемая вопросом часть должна соответствовать одному из обязательных тезисов.
Составь краткий эталонный ответ только из обязательных тезисов, без новых важных фактов.
Все поля пиши на русском языке.
Верни только JSON без Markdown и дополнительного текста.
Используй только ключи question, key_points и reference_answer.
Поля question и reference_answer должны быть строками, key_points должен быть массивом строк.
Все поля должны быть непустыми."""
_REPAIR_PROMPT = """Предыдущий ответ не прошёл проверку JSON.
Исправь только JSON-структуру, сохранив вопрос, тезисы и эталонный ответ.
Верни полный корректный JSON без Markdown и пояснений."""


class LlmGroundedRecallExerciseGenerator:
    def __init__(
        self,
        provider: LlmProvider,
        temperature: float = 0.2,
        max_tokens: int = 1000,
    ) -> None:
        if temperature < 0 or max_tokens <= 0:
            raise ValueError("Temperature must be non-negative and max tokens must be positive")

        self._provider = provider
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(
        self,
        note_title: str,
        markdown_content: str,
        question: str | None = None,
    ) -> RecallExercise:
        question_instruction = (
            f"Используй этот вопрос без изменений:\n{question}"
            if question is not None
            else "Сформулируй вопрос самостоятельно."
        )
        messages = [
            {
                "role": "system",
                "content": _SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Название заметки:\n{note_title}\n\n"
                    f"{question_instruction}\n\n"
                    f"Содержимое заметки:\n{markdown_content}"
                ),
            },
        ]
        content = self._request_content(messages)

        try:
            return RecallExercise.model_validate_json(content)
        except ValidationError:
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": content,
                    },
                    {
                        "role": "user",
                        "content": _REPAIR_PROMPT,
                    },
                ]
            )
            repaired_content = self._request_content(messages)

            return RecallExercise.model_validate_json(repaired_content)

    def _request_content(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        response = self._provider.generate(
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            json_schema=RecallExercise.model_json_schema(),
        )

        return response.text

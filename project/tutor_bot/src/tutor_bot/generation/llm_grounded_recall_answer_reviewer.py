from pydantic import ValidationError

from tutor_bot.application.recall_answer_review import RecallAnswerReview
from tutor_bot.application.recall_exercise import RecallExercise
from tutor_bot.generation.llm_provider import LlmProvider


_SYSTEM_PROMPT = """Ты проверяешь ответ пользователя в упражнении Active Recall.
Оценивай ответ только по вопросу, обязательным тезисам и эталонному ответу.
Не добавляй критерии и факты из внешних знаний.
covered_points должны содержать только тезисы, которые пользователь действительно раскрыл.
missing_points должны содержать обязательные тезисы, отсутствующие в ответе.
errors должны содержать только фактические противоречия эталону.
Используй correct только если missing_points и errors пусты.
Используй partially_correct, если раскрыта часть тезисов и нет ключевых ошибок.
Если covered_points и missing_points не пусты, а errors пуст, используй partially_correct.
Используй incorrect, если обязательные тезисы не раскрыты или есть ключевые ошибки.
Все элементы списков и feedback пиши на русском языке.
Не добавляй маркеры списков или нумерацию внутрь строк.
Не повторяй один смысл в нескольких формулировках.
Верни только JSON без Markdown и дополнительного текста.
Используй только ключи verdict, covered_points, missing_points, errors и feedback.
Поле feedback должно содержать непустую рекомендацию пользователю."""
_REPAIR_PROMPT = """Предыдущий ответ не прошёл проверку JSON.
Исправь только JSON-структуру, сохранив смысл оценки.
Верни полный корректный JSON без Markdown и пояснений."""


class LlmGroundedRecallAnswerReviewer:
    def __init__(
        self,
        provider: LlmProvider,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> None:
        if temperature < 0 or max_tokens <= 0:
            raise ValueError("Temperature must be non-negative and max tokens must be positive")

        self._provider = provider
        self._temperature = temperature
        self._max_tokens = max_tokens

    def review(
        self,
        exercise: RecallExercise,
        student_answer: str,
    ) -> RecallAnswerReview:
        key_points = "\n".join(f"- {key_point}" for key_point in exercise.key_points)
        messages = [
            {
                "role": "system",
                "content": _SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Вопрос:\n{exercise.question}\n\n"
                    f"Обязательные тезисы:\n{key_points}\n\n"
                    f"Эталонный ответ:\n{exercise.reference_answer}\n\n"
                    f"Ответ пользователя:\n{student_answer}"
                ),
            },
        ]
        content = self._request_content(messages)

        try:
            return RecallAnswerReview.model_validate_json(content)
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

            return RecallAnswerReview.model_validate_json(repaired_content)

    def _request_content(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        response = self._provider.generate(
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            json_schema=RecallAnswerReview.model_json_schema(),
        )

        return response.text

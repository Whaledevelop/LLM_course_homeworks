from ollama import Client

from tutor_bot.application.recall_answer_review import RecallAnswerReview
from tutor_bot.application.recall_exercise import RecallExercise


_DEFAULT_MODEL_NAME = "qwen3.5:9b"
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


class OllamaGroundedRecallAnswerReviewer:
    def __init__(
        self,
        base_url: str,
        model_name: str = _DEFAULT_MODEL_NAME,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout_seconds: float = 120.0,
        think: bool = False,
    ) -> None:
        if temperature < 0 or max_tokens <= 0 or timeout_seconds <= 0:
            raise ValueError("Temperature must be non-negative and limits must be positive")

        self._client = Client(
            host=base_url.removesuffix("/v1").rstrip("/"),
            timeout=timeout_seconds,
        )
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._think = think

    def review(
        self,
        exercise: RecallExercise,
        student_answer: str,
    ) -> RecallAnswerReview:
        key_points = "\n".join(f"- {key_point}" for key_point in exercise.key_points)

        response = self._client.chat(
            model=self._model_name,
            messages=[
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
            ],
            options={
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
            format=RecallAnswerReview.model_json_schema(),
            think=self._think,
        )

        content = response.message.content

        if not content:
            raise RuntimeError("Ollama returned an empty recall answer review")

        return RecallAnswerReview.model_validate_json(content)

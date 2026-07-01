from ollama import Client

from tutor_bot.application.recall_exercise import RecallExercise


_DEFAULT_MODEL_NAME = "qwen3.5:9b"
_SYSTEM_PROMPT = """Ты создаёшь упражнение Active Recall по одной учебной заметке.
Используй только информацию из переданной заметки и не добавляй внешние факты.
Сформулируй один сфокусированный открытый вопрос по одной основной теме заметки.
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


class OllamaGroundedRecallExerciseGenerator:
    def __init__(
        self,
        base_url: str,
        model_name: str = _DEFAULT_MODEL_NAME,
        temperature: float = 0.2,
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

    def generate(
        self,
        note_title: str,
        markdown_content: str,
    ) -> RecallExercise:
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
                        f"Название заметки:\n{note_title}\n\n"
                        f"Содержимое заметки:\n{markdown_content}"
                    ),
                },
            ],
            options={
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
            format=RecallExercise.model_json_schema(),
            think=self._think,
        )

        content = response.message.content

        if not content:
            raise RuntimeError("Ollama returned an empty recall exercise")

        return RecallExercise.model_validate_json(content)

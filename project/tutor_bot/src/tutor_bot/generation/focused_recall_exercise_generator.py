from pydantic import ValidationError

from tutor_bot.application.recall_exercise import RecallExercise
from tutor_bot.application.vacancy_study_target import VacancyStudyTarget
from tutor_bot.generation.llm_provider import LlmProvider


_SYSTEM_PROMPT = """Создай одно упражнение Active Recall для подготовки к собеседованию.
Используй только информацию из заметки. Вопрос должен проверять конкретное требование вакансии и не уходить в другие темы заметки.
Сформулируй открытый вопрос, 2-5 обязательных тезисов и краткий эталонный ответ.
Верни только JSON с ключами question, key_points и reference_answer. Все поля пиши на русском языке."""


class FocusedRecallExerciseGenerator:
    def __init__(self, provider: LlmProvider) -> None:
        self._provider = provider

    def generate(
        self,
        target: VacancyStudyTarget,
        markdown_content: str,
    ) -> RecallExercise:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Тема: {target.topic}\n"
                    f"Ожидаемые знания: {target.expected_knowledge}\n"
                    f"Заметка: {target.note_title}\n\n{markdown_content}"
                ),
            },
        ]
        response = self._provider.generate(
            messages=messages,
            temperature=0.2,
            max_tokens=1000,
            json_schema=RecallExercise.model_json_schema(),
        )

        try:
            return RecallExercise.model_validate_json(response.text)
        except ValidationError:
            messages.extend(
                [
                    {"role": "assistant", "content": response.text},
                    {
                        "role": "user",
                        "content": "Исправь структуру. Верни только корректный JSON по схеме.",
                    },
                ]
            )
            repaired_response = self._provider.generate(
                messages=messages,
                temperature=0.0,
                max_tokens=1000,
                json_schema=RecallExercise.model_json_schema(),
            )

            return RecallExercise.model_validate_json(repaired_response.text)

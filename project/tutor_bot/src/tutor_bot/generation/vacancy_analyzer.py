from pydantic import ValidationError

from tutor_bot.application.vacancy_analysis import VacancyAnalysis
from tutor_bot.generation.llm_provider import LlmProvider


_SYSTEM_PROMPT = """Проанализируй текст вакансии и выдели знания и навыки, которые будут проверяться на собеседовании.
Не добавляй требования, которых нет в тексте. Объединяй повторы и близкие формулировки.
title должен содержать короткое название вакансии.
Для каждого requirements укажи topic, expected_knowledge и evidence.
topic - короткое название темы, expected_knowledge - что кандидат должен знать или уметь, evidence - краткая цитата или точная формулировка из вакансии.
Верни только JSON по переданной схеме. Все содержательные поля пиши на языке вакансии."""


class VacancyAnalyzer:
    def __init__(self, provider: LlmProvider) -> None:
        self._provider = provider

    def analyze(self, vacancy_text: str) -> VacancyAnalysis:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": vacancy_text},
        ]
        response = self._provider.generate(
            messages=messages,
            temperature=0.0,
            max_tokens=4000,
            json_schema=VacancyAnalysis.model_json_schema(),
        )

        try:
            return VacancyAnalysis.model_validate_json(response.text)
        except ValidationError:
            messages.extend(
                [
                    {"role": "assistant", "content": response.text},
                    {
                        "role": "user",
                        "content": "Исправь структуру ответа. Верни только корректный JSON по схеме.",
                    },
                ]
            )
            repaired_response = self._provider.generate(
                messages=messages,
                temperature=0.0,
                max_tokens=4000,
                json_schema=VacancyAnalysis.model_json_schema(),
            )

            return VacancyAnalysis.model_validate_json(repaired_response.text)

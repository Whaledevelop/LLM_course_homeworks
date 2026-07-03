from pydantic import ValidationError

from tutor_bot.application.assignment_review import AssignmentReview
from tutor_bot.generation.llm_provider import LlmProvider
from tutor_bot.retrieval.context_gate_result import ContextGateResult


_SYSTEM_PROMPT = """Ты проверяешь учебное задание по локальным заметкам.
Оценивай ответ ученика только по переданному контексту и формулировке задания.
Не добавляй требования и факты из внешних знаний.
Отдельно перечисли правильные тезисы, фактические ошибки и пропущенные важные тезисы.
Вердикт correct используй только для полного ответа без существенных ошибок.
Вердикт partially_correct используй для ответа с верными тезисами, но заметными пропусками.
Вердикт incorrect используй для ответа с ключевыми фактическими ошибками.
Если errors или missing_points не пусты, verdict не может быть correct.
Если missing_points не пуст, но ключевых ошибок нет, используй partially_correct.
Все элементы списков и обратную связь пиши на русском языке.
Не повторяй один смысл в нескольких формулировках.
Верни только JSON, соответствующий переданной схеме, без Markdown и дополнительного текста.
Используй только ключи verdict, correct_points, errors, missing_points и feedback.
Поле feedback должно содержать непустую итоговую рекомендацию ученику.
Пиши конкретно и без лишней похвалы."""
_REPAIR_PROMPT = """Предыдущий ответ не прошёл проверку JSON.
Исправь только JSON-структуру, сохранив смысл оценки.
Верни полный корректный JSON без Markdown и пояснений."""


class OllamaGroundedAssignmentReviewer:
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
        assignment_text: str,
        student_answer: str,
        context: ContextGateResult,
    ) -> AssignmentReview:
        if not context.has_sufficient_context:
            return AssignmentReview(
                verdict="insufficient_context",
                correct_points=(),
                errors=(),
                missing_points=(),
                feedback="В заметках недостаточно информации для проверки этого задания.",
            )

        messages = [
            {
                "role": "system",
                "content": _SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": self._build_user_prompt(
                    assignment_text,
                    student_answer,
                    context,
                ),
            },
        ]
        content = self._request_content(messages)

        try:
            return AssignmentReview.model_validate_json(content)
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

            return AssignmentReview.model_validate_json(repaired_content)

    def _request_content(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        response = self._provider.generate(
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            json_schema=AssignmentReview.model_json_schema(),
        )

        return response.text

    def _build_user_prompt(
        self,
        assignment_text: str,
        student_answer: str,
        context: ContextGateResult,
    ) -> str:
        sources = []

        for source_number, result in enumerate(
            context.selected_results,
            start=1,
        ):
            sources.append(
                "\n".join(
                    [
                        f'<source id="{source_number}">',
                        f"Название: {result.chunk.note_title}",
                        result.chunk.text,
                        "</source>",
                    ]
                )
            )

        joined_sources = "\n\n".join(sources)

        return (
            f"Задание:\n{assignment_text}\n\n"
            f"Ответ ученика:\n{student_answer}\n\n"
            f"Контекст:\n{joined_sources}"
        )

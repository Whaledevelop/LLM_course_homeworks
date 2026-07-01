from openai import OpenAI

from tutor_bot.application.assignment_review import AssignmentReview
from tutor_bot.retrieval.context_gate_result import ContextGateResult


_DEFAULT_MODEL_NAME = "qwen2.5:3b"
_SYSTEM_PROMPT = """Ты проверяешь учебное задание по локальным заметкам.
Оценивай ответ ученика только по переданному контексту и формулировке задания.
Не добавляй требования и факты из внешних знаний.
Отдельно перечисли правильные тезисы, фактические ошибки и пропущенные важные тезисы.
Вердикт correct используй только для полного ответа без существенных ошибок.
Вердикт partially_correct используй для ответа с верными тезисами, но заметными пропусками.
Вердикт incorrect используй для ответа с ключевыми фактическими ошибками.
Все элементы списков и обратную связь пиши на русском языке.
Не повторяй один смысл в нескольких формулировках.
Пиши конкретно и без лишней похвалы."""


class OllamaGroundedAssignmentReviewer:
    def __init__(
        self,
        base_url: str,
        model_name: str = _DEFAULT_MODEL_NAME,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout_seconds: float = 120.0,
    ) -> None:
        if temperature < 0 or max_tokens <= 0 or timeout_seconds <= 0:
            raise ValueError("Temperature must be non-negative and limits must be positive")

        self._client = OpenAI(
            base_url=base_url,
            api_key="ollama",
            timeout=timeout_seconds,
        )
        self._model_name = model_name
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

        completion = self._client.chat.completions.create(
            model=self._model_name,
            messages=[
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
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "assignment_review",
                    "strict": True,
                    "schema": AssignmentReview.model_json_schema(),
                },
            },
        )

        content = completion.choices[0].message.content

        if content is None:
            raise RuntimeError("Ollama returned an empty assignment review")

        return AssignmentReview.model_validate_json(content)

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

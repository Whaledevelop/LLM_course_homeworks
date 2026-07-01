from ollama import Client

from tutor_bot.retrieval.context_gate_result import ContextGateResult


_DEFAULT_MODEL_NAME = "qwen2.5:3b"
_NO_CONTEXT_RESPONSE = "В заметках недостаточно информации для ответа на этот вопрос."
_SYSTEM_PROMPT = """Ты локальный учебный ассистент.
Отвечай только на основе переданных фрагментов заметок.
Не добавляй факты из внешних знаний и не выдумывай детали.
Если контекста недостаточно, прямо сообщи об этом.
После утверждений указывай источники в формате [1], [2].
Отвечай на русском языке, кратко и по существу."""


class OllamaGroundedAnswerGenerator:
    def __init__(
        self,
        base_url: str,
        model_name: str = _DEFAULT_MODEL_NAME,
        temperature: float = 0.0,
        max_tokens: int = 800,
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
        question: str,
        context: ContextGateResult,
    ) -> str:
        if not context.has_sufficient_context:
            return _NO_CONTEXT_RESPONSE

        response = self._client.chat(
            model=self._model_name,
            messages=[
                {
                    "role": "system",
                    "content": _SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": self._build_user_prompt(
                        question,
                        context,
                    ),
                },
            ],
            options={
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
            think=self._think,
        )

        answer = response.message.content

        if not answer:
            raise RuntimeError("Ollama returned an empty answer")

        source_list = self._build_source_list(context)

        return f"{answer.strip()}\n\nИсточники:\n{source_list}"

    def _build_user_prompt(
        self,
        question: str,
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
                        f"Путь: {result.chunk.relative_path.as_posix()}",
                        result.chunk.text,
                        "</source>",
                    ]
                )
            )

        joined_sources = "\n\n".join(sources)

        return f"Вопрос:\n{question}\n\nКонтекст:\n{joined_sources}"

    def _build_source_list(
        self,
        context: ContextGateResult,
    ) -> str:
        source_lines = [
            (
                f"[{source_number}] {result.chunk.note_title} — "
                f"`{result.chunk.relative_path.as_posix()}`"
            )
            for source_number, result in enumerate(
                context.selected_results,
                start=1,
            )
        ]

        return "\n".join(source_lines)

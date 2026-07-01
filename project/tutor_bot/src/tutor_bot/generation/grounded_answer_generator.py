from typing import Protocol

from tutor_bot.retrieval.context_gate_result import ContextGateResult


class GroundedAnswerGenerator(Protocol):
    def generate(
        self,
        question: str,
        context: ContextGateResult,
    ) -> str: ...

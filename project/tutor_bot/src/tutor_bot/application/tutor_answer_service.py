from tutor_bot.application.tutor_answer import TutorAnswer
from tutor_bot.generation.grounded_answer_generator import GroundedAnswerGenerator
from tutor_bot.retrieval.hybrid_search_service import HybridSearchService
from tutor_bot.retrieval.reranker_context_gate import RerankerContextGate


class TutorAnswerService:
    def __init__(
        self,
        search_service: HybridSearchService,
        context_gate: RerankerContextGate,
        answer_generator: GroundedAnswerGenerator,
        search_limit: int = 10,
    ) -> None:
        if search_limit <= 0:
            raise ValueError("Search limit must be positive")

        self._search_service = search_service
        self._context_gate = context_gate
        self._answer_generator = answer_generator
        self._search_limit = search_limit

    def answer(
        self,
        question: str,
    ) -> TutorAnswer:
        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError("Question must not be empty")

        search_results = self._search_service.search(
            normalized_question,
            limit=self._search_limit,
        )

        context = self._context_gate.select(search_results)
        generated_answer = self._answer_generator.generate(
            normalized_question,
            context,
        )

        return TutorAnswer(
            question=normalized_question,
            answer=generated_answer,
            context=context,
        )

import json

from tutor_bot.application.vacancy_match import VacancyMatch
from tutor_bot.application.vacancy_match_decision import VacancyMatchDecision
from tutor_bot.application.vacancy_requirement import VacancyRequirement
from tutor_bot.generation.llm_provider import LlmProvider
from tutor_bot.retrieval.hybrid_search_service import HybridSearchService


_SYSTEM_PROMPT = """Определи, покрывает ли одна из заметок требование вакансии.
Совпадение должно быть смысловым, а не обязательно дословным. Учитывай синонимы, сокращения и альтернативные названия технологии.
Выбирай заметку только если ее фрагмент содержит знания, достаточные для подготовки по требованию.
Если надежного совпадения нет, верни matched_note_title null и confidence от 0 до 0.49.
Если совпадение есть, верни точное note_title из кандидатов и confidence от 0.5 до 1.
Верни только JSON по схеме."""


class VacancyMatchingService:
    def __init__(
        self,
        search_service: HybridSearchService,
        provider: LlmProvider,
        candidate_limit: int = 5,
    ) -> None:
        self._search_service = search_service
        self._provider = provider
        self._candidate_limit = candidate_limit

    def match(self, requirement: VacancyRequirement) -> VacancyMatch:
        query = f"{requirement.topic}. {requirement.expected_knowledge}"
        search_results = self._search_service.search(query, limit=self._candidate_limit)

        if not search_results:
            return VacancyMatch(requirement=requirement, confidence=0.0)

        candidates = [
            {
                "note_title": result.chunk.note_title,
                "fragment": result.chunk.text,
            }
            for result in search_results
        ]
        response = self._provider.generate(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Требование: {query}\n\n"
                        f"Кандидаты: {json.dumps(candidates, ensure_ascii=False)}"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=300,
            json_schema=VacancyMatchDecision.model_json_schema(),
        )
        decision = VacancyMatchDecision.model_validate_json(response.text)
        matched_result = next(
            (
                result
                for result in search_results
                if result.chunk.note_title == decision.matched_note_title
            ),
            None,
        )

        if matched_result is None or decision.confidence < 0.5:
            return VacancyMatch(requirement=requirement, confidence=decision.confidence)

        return VacancyMatch(
            requirement=requirement,
            note_id=matched_result.chunk.note_id,
            note_title=matched_result.chunk.note_title,
            confidence=decision.confidence,
        )

    def match_all(
        self,
        requirements: tuple[VacancyRequirement, ...],
    ) -> tuple[VacancyMatch, ...]:
        return tuple(self.match(requirement) for requirement in requirements)

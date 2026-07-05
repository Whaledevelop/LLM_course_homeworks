from random import SystemRandom

from tutor_bot.application.active_recall_service import ActiveRecallService
from tutor_bot.application.note_query_service import NoteQueryService
from tutor_bot.application.recall_session import RecallSession
from tutor_bot.application.vacancy_match import VacancyMatch
from tutor_bot.application.vacancy_study_session import VacancyStudySession
from tutor_bot.application.vacancy_study_target import VacancyStudyTarget
from tutor_bot.generation.focused_recall_exercise_generator import FocusedRecallExerciseGenerator


class VacancyPreparationService:
    def __init__(
        self,
        note_query_service: NoteQueryService,
        exercise_generator: FocusedRecallExerciseGenerator,
        active_recall_service: ActiveRecallService,
    ) -> None:
        self._note_query_service = note_query_service
        self._exercise_generator = exercise_generator
        self._active_recall_service = active_recall_service

    def create_study_session(
        self,
        vacancy_title: str,
        matches: tuple[VacancyMatch, ...],
        question_count: int,
    ) -> VacancyStudySession:
        covered_matches = [vacancy_match for vacancy_match in matches if vacancy_match.is_covered]

        if question_count <= 0 or question_count > len(covered_matches):
            raise ValueError("Question count must fit covered vacancy requirements")

        selected_matches = SystemRandom().sample(covered_matches, question_count)
        targets = tuple(self._create_target(vacancy_match) for vacancy_match in selected_matches)

        return VacancyStudySession(
            vacancy_title=vacancy_title,
            targets=targets,
            current_index=0,
            current_exercise=self._create_session(vacancy_title, targets[0]),
        )

    def review_study_answer(
        self,
        study_session: VacancyStudySession,
        student_answer: str,
    ) -> VacancyStudySession:
        return self._active_recall_service.review_study_answer(study_session, student_answer)

    def imitate_study_answer(
        self,
        study_session: VacancyStudySession,
    ) -> VacancyStudySession:
        return self._active_recall_service.imitate_study_answer(study_session)

    def advance_study_session(
        self,
        study_session: VacancyStudySession,
    ) -> VacancyStudySession:
        if study_session.answered_count <= study_session.current_index:
            raise ValueError("Current exercise must be completed before advancing")

        next_index = study_session.current_index + 1

        if next_index >= study_session.total_count:
            return study_session

        return study_session.model_copy(
            update={
                "current_index": next_index,
                "current_exercise": self._create_session(
                    study_session.vacancy_title,
                    study_session.targets[next_index],
                ),
            },
        )

    def _create_target(self, vacancy_match: VacancyMatch) -> VacancyStudyTarget:
        if vacancy_match.note_id is None or vacancy_match.note_title is None:
            raise ValueError("Vacancy requirement is not covered")

        return VacancyStudyTarget(
            requirement_id=vacancy_match.requirement.id,
            topic=vacancy_match.requirement.topic,
            expected_knowledge=vacancy_match.requirement.expected_knowledge,
            note_id=vacancy_match.note_id,
            note_title=vacancy_match.note_title,
        )

    def _create_session(
        self,
        vacancy_title: str,
        target: VacancyStudyTarget,
    ) -> RecallSession:
        note = self._note_query_service.get_note(target.note_id)
        exercise = self._exercise_generator.generate(target, note.markdown_content)

        return RecallSession(
            note_id=note.id,
            note_title=note.title,
            source_markdown=note.markdown_content,
            context_title=vacancy_title,
            focus_topic=target.topic,
            exercise=exercise,
        )

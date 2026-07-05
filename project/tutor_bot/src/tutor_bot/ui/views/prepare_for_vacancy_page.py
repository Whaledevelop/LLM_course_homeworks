import streamlit as st
from httpx import HTTPError
from pydantic import ValidationError

from tutor_bot.application.vacancy import Vacancy
from tutor_bot.application.vacancy_match import VacancyMatch
from tutor_bot.application.vacancy_matching_service import VacancyMatchingService
from tutor_bot.application.vacancy_preparation_service import VacancyPreparationService
from tutor_bot.generation.vacancy_analyzer import VacancyAnalyzer
from tutor_bot.infrastructure.vacancy_repository import VacancyRepository
from tutor_bot.ui.views.active_recall_session_view import render_active_recall_session


_SESSION_KEY = "vacancy_study_session"
_SELECTED_VACANCY_KEY = "selected_vacancy_id"
_MATCHES_KEY = "selected_vacancy_matches"
_MATCHES_VACANCY_KEY = "matched_vacancy_id"


def interrupt_vacancy_session() -> None:
    st.session_state.pop(_SESSION_KEY, None)
    _clear_matches()


def render_prepare_for_vacancy_page(
    repository: VacancyRepository,
    analyzer: VacancyAnalyzer,
    matching_service: VacancyMatchingService,
    preparation_service: VacancyPreparationService,
) -> None:
    st.header("Prepare for Vacancy")
    _render_uploader(repository, analyzer)
    vacancies = repository.list_vacancies()

    if not vacancies:
        st.info("Загрузите PDF вакансии, чтобы начать подготовку.")

        return

    selected_vacancy = _render_vacancy_selector(vacancies)
    matches = _get_matches(selected_vacancy, matching_service)
    _render_coverage(matches)
    study_session = st.session_state.get(_SESSION_KEY)

    if study_session is None:
        _render_start_controls(selected_vacancy, matches, preparation_service)

        return

    st.divider()
    render_active_recall_session(
        preparation_service,
        study_session,
        _SESSION_KEY,
        False,
        False,
    )


def _render_uploader(
    repository: VacancyRepository,
    analyzer: VacancyAnalyzer,
) -> None:
    with st.form("vacancy-upload-form", clear_on_submit=True):
        uploaded_file = st.file_uploader("PDF вакансии", type=("pdf",))
        submitted = st.form_submit_button("Загрузить и проанализировать", type="primary")

    if not submitted:
        return

    if uploaded_file is None:
        st.error("Выберите PDF-файл.")

        return

    try:
        pdf_content = uploaded_file.getvalue()
        vacancy_text = repository.extract_text(pdf_content)

        with st.spinner("Анализирую требования вакансии..."):
            analysis = analyzer.analyze(vacancy_text)
            vacancy = repository.save_pdf(uploaded_file.name, pdf_content, analysis)

        st.session_state[_SELECTED_VACANCY_KEY] = str(vacancy.id)
        _clear_matches()
        interrupt_vacancy_session()
        st.success(f"Вакансия «{vacancy.title}» сохранена.")
    except (HTTPError, RuntimeError, ValueError, ValidationError) as error:
        st.error(f"Не удалось обработать вакансию: {error}")


def _render_vacancy_selector(vacancies: tuple[Vacancy, ...]) -> Vacancy:
    vacancy_by_id = {str(vacancy.id): vacancy for vacancy in vacancies}
    selected_id = st.session_state.get(_SELECTED_VACANCY_KEY)

    if selected_id not in vacancy_by_id:
        selected_id = str(vacancies[0].id)

    selected_id = st.selectbox(
        "Загруженные вакансии",
        options=tuple(vacancy_by_id),
        index=tuple(vacancy_by_id).index(selected_id),
        format_func=lambda vacancy_id: vacancy_by_id[vacancy_id].title,
        key=_SELECTED_VACANCY_KEY,
        on_change=_on_vacancy_changed,
    )

    return vacancy_by_id[selected_id]


def _get_matches(
    vacancy: Vacancy,
    matching_service: VacancyMatchingService,
) -> tuple[VacancyMatch, ...]:
    cached_vacancy_id = st.session_state.get(_MATCHES_VACANCY_KEY)

    if cached_vacancy_id == str(vacancy.id):
        return st.session_state[_MATCHES_KEY]

    with st.spinner("Сопоставляю требования с текущей базой заметок..."):
        matches = matching_service.match_all(vacancy.requirements)

    st.session_state[_MATCHES_KEY] = matches
    st.session_state[_MATCHES_VACANCY_KEY] = str(vacancy.id)

    return matches


def _render_coverage(matches: tuple[VacancyMatch, ...]) -> None:
    covered_matches = [vacancy_match for vacancy_match in matches if vacancy_match.is_covered]
    missing_matches = [vacancy_match for vacancy_match in matches if not vacancy_match.is_covered]
    st.subheader("Покрытие требований")

    for vacancy_match in covered_matches:
        st.markdown(
            f"- **{vacancy_match.requirement.topic}** → {vacancy_match.note_title} "
            f"({vacancy_match.confidence:.0%})"
        )

    if missing_matches:
        missing_topics = ", ".join(
            vacancy_match.requirement.topic for vacancy_match in missing_matches
        )
        st.warning(f"В базе заметок не хватает тем: {missing_topics}")


def _render_start_controls(
    vacancy: Vacancy,
    matches: tuple[VacancyMatch, ...],
    preparation_service: VacancyPreparationService,
) -> None:
    covered_count = sum(vacancy_match.is_covered for vacancy_match in matches)

    if covered_count == 0:
        st.info("Тест недоступен: ни одно требование не покрыто заметками.")

        return

    question_count = st.slider(
        "Количество вопросов",
        min_value=1,
        max_value=covered_count,
        value=covered_count,
        key=f"vacancy_question_count_{vacancy.id}",
    )

    if st.button("Начать подготовку", type="primary"):
        try:
            with st.spinner("Генерирую первый вопрос..."):
                st.session_state[_SESSION_KEY] = preparation_service.create_study_session(
                    vacancy.title,
                    matches,
                    question_count,
                )
            st.rerun()
        except (HTTPError, RuntimeError, ValueError, ValidationError) as error:
            st.error(f"Не удалось начать тест: {error}")


def _on_vacancy_changed() -> None:
    _clear_matches()
    interrupt_vacancy_session()


def _clear_matches() -> None:
    st.session_state.pop(_MATCHES_KEY, None)
    st.session_state.pop(_MATCHES_VACANCY_KEY, None)

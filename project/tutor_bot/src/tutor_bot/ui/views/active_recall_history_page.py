import streamlit as st

from tutor_bot.application.active_recall_service import ActiveRecallService
from tutor_bot.application.recall_session_result import RecallSessionResult
from tutor_bot.ui.views.active_recall_session_view import VERDICT_LABELS


def render_active_recall_history_page(
    recall_service: ActiveRecallService,
) -> None:
    history = recall_service.get_history()

    if st.button(
        "Clear History",
        disabled=not history,
    ):
        recall_service.clear_history()
        st.rerun()

    if not history:
        st.info("История попыток пока отсутствует.")

        return

    _render_summary(history)
    _render_latest_attempts(history)


def _render_summary(
    history: tuple[RecallSessionResult, ...],
) -> None:
    verdict_counts = {
        verdict: sum(result.review.verdict == verdict and not result.imitated for result in history)
        for verdict in VERDICT_LABELS
    }
    columns = st.columns(5)
    columns[0].metric("Попыток", len(history))
    columns[1].metric("Правильных", verdict_counts["correct"])
    columns[2].metric("Частичных", verdict_counts["partially_correct"])
    columns[3].metric("Ошибочных", verdict_counts["incorrect"])
    columns[4].metric("Показан ответ", sum(result.imitated for result in history))


def _render_latest_attempts(
    history: tuple[RecallSessionResult, ...],
) -> None:
    st.subheader("Последние попытки")

    for result in reversed(history[-10:]):
        verdict_label = (
            "Эталонный ответ показан" if result.imitated else VERDICT_LABELS[result.review.verdict]
        )
        st.markdown(f"**{result.session.note_title}** — {verdict_label}")
        if result.session.context_title is not None:
            st.caption(
                f"Vacancy: {result.session.context_title} · Тема: {result.session.focus_topic}"
            )
        st.caption(result.session.exercise.question)

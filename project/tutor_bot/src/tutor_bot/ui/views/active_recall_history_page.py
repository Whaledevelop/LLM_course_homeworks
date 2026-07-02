import streamlit as st

from tutor_bot.application.active_recall_service import ActiveRecallService
from tutor_bot.application.recall_session_result import RecallSessionResult
from tutor_bot.ui.views.active_recall_session_view import VERDICT_LABELS


def render_active_recall_history_page(
    recall_service: ActiveRecallService,
) -> None:
    history = recall_service.get_history()

    if not history:
        st.info("История попыток пока отсутствует.")

        return

    _render_summary(history)
    _render_latest_attempts(history)


def _render_summary(
    history: tuple[RecallSessionResult, ...],
) -> None:
    verdict_counts = {
        verdict: sum(result.review.verdict == verdict for result in history)
        for verdict in VERDICT_LABELS
    }
    columns = st.columns(4)
    columns[0].metric("Попыток", len(history))
    columns[1].metric("Правильных", verdict_counts["correct"])
    columns[2].metric("Частичных", verdict_counts["partially_correct"])
    columns[3].metric("Ошибочных", verdict_counts["incorrect"])


def _render_latest_attempts(
    history: tuple[RecallSessionResult, ...],
) -> None:
    st.subheader("Последние попытки")

    for result in reversed(history[-10:]):
        st.markdown(
            f"**{result.session.note_title}** — {VERDICT_LABELS[result.review.verdict]}"
        )
        st.caption(result.session.exercise.question)

import streamlit as st

from tutor_bot.application.observability_event_service import ObservabilityEventService
from tutor_bot.application.observability_statistics import ObservabilityStatistics


def render_observability_page(
    observability_event_service: ObservabilityEventService,
) -> None:
    statistics = observability_event_service.get_statistics()

    if statistics.total_events == 0:
        st.info("События пока не записаны.")

        return

    columns = st.columns(4)
    columns[0].metric("Событий", statistics.total_events)
    columns[1].metric("Сценариев", len(statistics.events_by_scenario))
    columns[2].metric("Статусов", len(statistics.events_by_status))
    columns[3].metric("Ошибок", statistics.events_by_status.get("failed", 0))

    _render_counts(
        "События по сценариям",
        statistics.events_by_scenario,
    )
    _render_counts(
        "События по типам",
        statistics.events_by_event_type,
    )
    _render_counts(
        "События по статусам",
        statistics.events_by_status,
    )
    _render_counts(
        "LLM модели",
        statistics.events_by_model,
    )
    _render_success_rates(statistics)
    _render_durations(statistics)
    _render_latest_errors(statistics)


def _render_counts(
    title: str,
    values: dict[str, int],
) -> None:
    st.subheader(title)
    st.dataframe(
        [
            {
                "name": name,
                "count": count,
            }
            for name, count in sorted(values.items())
        ],
        hide_index=True,
        use_container_width=True,
    )


def _render_durations(
    statistics: ObservabilityStatistics,
) -> None:
    if not statistics.average_duration_seconds_by_scenario:
        return

    st.subheader("Средняя длительность")
    st.dataframe(
        [
            {
                "scenario": scenario,
                "seconds": duration_seconds,
            }
            for scenario, duration_seconds in sorted(
                statistics.average_duration_seconds_by_scenario.items()
            )
        ],
        hide_index=True,
        use_container_width=True,
    )


def _render_success_rates(
    statistics: ObservabilityStatistics,
) -> None:
    if not statistics.success_rate_by_scenario:
        return

    st.subheader("Успешность сценариев")
    st.dataframe(
        [
            {
                "scenario": scenario,
                "success_rate_percent": success_rate,
            }
            for scenario, success_rate in sorted(
                statistics.success_rate_by_scenario.items()
            )
        ],
        hide_index=True,
        use_container_width=True,
    )


def _render_latest_errors(
    statistics: ObservabilityStatistics,
) -> None:
    if not statistics.latest_errors:
        return

    st.subheader("Последние ошибки")
    st.dataframe(
        [
            {
                "recorded_at": event.recorded_at.isoformat(),
                "scenario": event.scenario,
                "event_type": event.event_type,
                "error": event.error,
            }
            for event in statistics.latest_errors
        ],
        hide_index=True,
        use_container_width=True,
    )

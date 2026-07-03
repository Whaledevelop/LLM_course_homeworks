from pathlib import Path

import streamlit as st

from tutor_bot.infrastructure.active_database_service import (
    ActiveDatabaseService,
    DatabaseSummary,
)
from tutor_bot.ui.tutor_answer_service_factory import rebuild_database_search_index


def render_databases_page(database_service: ActiveDatabaseService) -> None:
    st.title("Базы данных")
    _render_create_database(database_service)
    summaries = database_service.list_summaries()

    if not summaries:
        st.info("Базы заметок еще не созданы.")

        return

    active_database = database_service.get_active()
    active_db_id = active_database.db_id if active_database is not None else summaries[0].db_id
    selected_db_id = st.selectbox(
        "Активная DB",
        options=[summary.db_id for summary in summaries],
        index=[summary.db_id for summary in summaries].index(active_db_id),
    )

    if selected_db_id != active_db_id:
        database_service.select(selected_db_id)
        st.cache_resource.clear()
        _clear_note_session_state()
        st.rerun()

    selected_summary = next(summary for summary in summaries if summary.db_id == selected_db_id)
    _render_summary(selected_summary)

    if st.button("Update index", type="primary", use_container_width=True):
        _update_index(database_service)

    st.subheader("Зарегистрированные DB")
    st.dataframe(
        [
            {
                "DB": summary.db_id,
                "Активна": "Да" if summary.is_active else "",
                "Заметок": summary.note_count,
                "В архиве": summary.archived_note_count,
                "Папка": str(summary.root_path),
            }
            for summary in summaries
        ],
        hide_index=True,
        use_container_width=True,
    )
    _render_groups(selected_summary)
    _render_delete_database(database_service, selected_summary)


def _render_create_database(database_service: ActiveDatabaseService) -> None:
    with st.expander("Добавить новую DB"):
        db_id = st.text_input("DB ID", key="databases_new_db_id")
        root_path = st.text_input("Папка с заметками", key="databases_new_db_path")

        if st.button("Создать и индексировать DB"):
            try:
                result = database_service.register(db_id.strip(), Path(root_path))
                st.success(
                    f"DB создана: добавлено {result.added}, перемещено {result.moved}, "
                    f"архивировано {result.archived}."
                )
                active_database = database_service.get_active()

                if active_database is not None:
                    with st.spinner("Перестроение retrieval index..."):
                        rebuild_database_search_index(
                            active_database.db_id,
                            str(active_database.root_path),
                        )

                st.cache_resource.clear()
                st.rerun()
            except Exception as error:
                st.error(str(error))


def _render_summary(summary: DatabaseSummary) -> None:
    columns = st.columns(3)
    columns[0].metric("Заметок", summary.note_count)
    columns[1].metric("Групп", len(summary.groups))
    columns[2].metric("В архиве", summary.archived_note_count)
    st.caption(f"Корневая папка: {summary.root_path}")


def _render_groups(summary: DatabaseSummary) -> None:
    st.subheader("Распределение по группам")

    if not summary.groups:
        st.info("В активной DB пока нет заметок.")

        return

    st.dataframe(
        [
            {"Группа": group, "Заметок": count}
            for group, count in sorted(
                summary.groups.items(),
                key=lambda item: (-item[1], item[0].casefold()),
            )
        ],
        hide_index=True,
        use_container_width=True,
    )


def _update_index(database_service: ActiveDatabaseService) -> None:
    try:
        with st.spinner("Обновление файлового index..."):
            result = database_service.update_active()
        active_database = database_service.get_active()
        st.success(
            f"Index обновлен: добавлено {result.added}, перемещено {result.moved}, "
            f"архивировано {result.archived}, восстановлено {result.restored}."
        )

        if active_database is not None:
            with st.spinner("Перестроение retrieval index..."):
                chunk_count = rebuild_database_search_index(
                    active_database.db_id,
                    str(active_database.root_path),
                )
            st.success(f"Retrieval index обновлен: {chunk_count} фрагментов.")

        st.cache_resource.clear()
    except Exception as error:
        st.error(str(error))


def _render_delete_database(
    database_service: ActiveDatabaseService,
    summary: DatabaseSummary,
) -> None:
    st.divider()
    with st.expander("Удалить DB из Tutor Bot"):
        st.warning(
            "Будут удалены только служебные данные Tutor Bot. Markdown-заметки останутся "
            "без изменений."
        )
        confirmation = st.text_input(
            f"Введите {summary.db_id} для подтверждения",
            key=f"delete_database_confirmation_{summary.db_id}",
        )

        if st.button(
            "Удалить DB",
            type="secondary",
            disabled=confirmation != summary.db_id,
        ):
            try:
                database_service.remove(summary.db_id)
                st.cache_resource.clear()
                _clear_note_session_state()
                st.rerun()
            except Exception as error:
                st.error(str(error))


def _clear_note_session_state() -> None:
    preserved_keys = {"selected_app_mode"}

    for key in tuple(st.session_state):
        if key not in preserved_keys:
            st.session_state.pop(key, None)

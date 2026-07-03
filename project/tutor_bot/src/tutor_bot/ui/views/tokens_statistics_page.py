import streamlit as st

from tutor_bot.ui.llm_session_state import get_usage


def render_tokens_statistics_page() -> None:
    usage_items = get_usage()
    prompt_tokens = sum(int(item["prompt_tokens"]) for item in usage_items)
    completion_tokens = sum(int(item["completion_tokens"]) for item in usage_items)
    total_tokens = sum(int(item["total_tokens"]) for item in usage_items)
    columns = st.columns(3)
    columns[0].metric("Prompt tokens", prompt_tokens)
    columns[1].metric("Completion tokens", completion_tokens)
    columns[2].metric("Total tokens", total_tokens)

    tokens_by_model: dict[str, int] = {}

    for item in usage_items:
        key = f"{item['provider']} / {item['model']}"
        tokens_by_model[key] = tokens_by_model.get(key, 0) + int(item["total_tokens"])

    if not tokens_by_model:
        st.info("Статистика токенов пока отсутствует.")

        return

    st.dataframe(
        [
            {"provider / model": key, "total_tokens": tokens}
            for key, tokens in sorted(tokens_by_model.items())
        ],
        hide_index=True,
        width="stretch",
    )

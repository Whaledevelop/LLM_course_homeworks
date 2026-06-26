import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

from extractors import RuleBasedNewsExtractor
from schemas import NewsDialog


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="")
    args, _ = parser.parse_known_args()

    st.set_page_config(page_title="News Dialog IE", layout="wide")
    st.title("Извлечение сущностей и событий из новостных диалогов")

    default_text = load_default_text(args.data)
    text = st.text_area("Диалог", default_text, height=260)
    extractor = RuleBasedNewsExtractor()
    dialog = NewsDialog(dialog_id="ui-demo", source="manual", text=text)
    result = extractor.extract_batch([dialog])[0]

    left, right = st.columns(2)
    with left:
        st.subheader("Сущности")
        st.dataframe(to_dataframe(result.entities), use_container_width=True)
    with right:
        st.subheader("События")
        st.dataframe(to_dataframe(result.events), use_container_width=True)

    st.subheader("Отношения")
    st.dataframe(pd.DataFrame(result.relations), use_container_width=True)

    st.subheader("JSON")
    st.json(asdict(result))


def load_default_text(data_path: str) -> str:
    if data_path:
        path = Path(data_path)
        if path.exists():
            with path.open("r", encoding="utf-8") as file:
                row = json.loads(file.readline())

            return row["text"]

    return (
        "User: Reuters reported that President Joe Biden met NATO Secretary General Jens Stoltenberg "
        "in Washington on April 4, 2024. What was the impact?\n"
        "Assistant: The meeting focused on Ukraine aid and strengthened NATO coordination before the summit."
    )


def to_dataframe(items) -> pd.DataFrame:
    rows = [asdict(item) for item in items]
    if not rows:
        return pd.DataFrame(columns=["label", "value", "start", "end", "confidence"])

    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

from extractors import RuleBasedNewsExtractor, SpacyNewsExtractor, TransformersJsonExtractor
from schemas import NewsDialog


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="")
    args, _ = parser.parse_known_args()

    st.set_page_config(page_title="News Dialog IE", layout="wide")
    st.title("Извлечение сущностей и событий из новостных диалогов")

    default_text = load_default_text(args.data)
    text = st.text_area("Диалог", default_text, height=260)
    profile = st.selectbox("Extractor", ["rules", "spacy", "qwen-fp16", "qwen-int8", "gemma-fp16", "gemma-int8"])
    try:
        extractor = load_extractor(profile)
    except Exception as error:
        st.error(f"Не удалось загрузить extractor: {error}")
        return
    dialog = NewsDialog(dialog_id="ui-demo", source="manual", text=text)
    started_at = time.perf_counter()
    result = extractor.extract_batch([dialog])[0]
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    st.metric("Время обработки", f"{elapsed_ms:.1f} ms")
    if result.error:
        st.error(result.error)

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


@st.cache_resource
def load_extractor(profile: str):
    if profile == "rules":
        return RuleBasedNewsExtractor()
    if profile == "spacy":
        return SpacyNewsExtractor()
    alias, precision_mode = profile.rsplit("-", maxsplit=1)
    models = {
        "qwen": "Qwen/Qwen3-1.7B",
        "gemma": "google/gemma-2-2b-it",
    }

    return TransformersJsonExtractor(models[alias], precision_mode)


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

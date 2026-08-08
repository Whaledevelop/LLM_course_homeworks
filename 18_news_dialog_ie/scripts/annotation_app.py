import csv
import json
from pathlib import Path

import pandas as pd
import streamlit as st


LABELS = ["PERSON", "ORG", "LOC", "EVENT", "DATE", "IMPACT", "SOURCE"]
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
DIALOGS_PATH = DATA_DIR / "news_dialogs.jsonl"
TEMPLATE_PATH = DATA_DIR / "gold_annotation_template.csv"
GOLD_PATH = DATA_DIR / "gold_annotations.csv"
PROGRESS_PATH = DATA_DIR / "annotation_progress.json"


def load_dialogs(path: Path) -> dict[str, str]:
    dialogs = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            dialog_id = str(row["dialog_id"])
            dialogs[dialog_id] = str(row["text"])

    return dialogs


def load_template_ids(path: Path) -> list[str]:
    dialog_ids = []
    seen = set()
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "dialog_id" not in reader.fieldnames:
            raise ValueError("Annotation template must contain a dialog_id column.")
        for row in reader:
            dialog_id = row["dialog_id"].strip()
            if not dialog_id or dialog_id in seen:
                continue
            seen.add(dialog_id)
            dialog_ids.append(dialog_id)

    return dialog_ids


def load_annotations(path: Path, allowed_dialog_ids: set[str]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    annotations = []
    seen = set()
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or set(reader.fieldnames) != {"dialog_id", "label", "value"}:
            raise ValueError("Gold CSV must contain dialog_id, label and value columns.")
        for row in reader:
            dialog_id = row["dialog_id"].strip()
            label = row["label"].strip().upper()
            value = row["value"].strip()
            key = (dialog_id, label, value)
            if dialog_id not in allowed_dialog_ids or label not in LABELS or not value or key in seen:
                continue
            seen.add(key)
            annotations.append({"dialog_id": dialog_id, "label": label, "value": value})

    return annotations


def save_annotations(path: Path, annotations: list[dict[str, str]]) -> None:
    rows = [row for row in annotations if row["dialog_id"].strip() and row["label"] in LABELS and row["value"].strip()]
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["dialog_id", "label", "value"])
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def load_reviewed(path: Path, allowed_dialog_ids: set[str]) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    reviewed = payload.get("reviewed_dialog_ids", [])

    return {str(dialog_id) for dialog_id in reviewed if str(dialog_id) in allowed_dialog_ids}


def save_reviewed(path: Path, reviewed_dialog_ids: set[str], ordered_dialog_ids: list[str]) -> None:
    ordered_reviewed = [dialog_id for dialog_id in ordered_dialog_ids if dialog_id in reviewed_dialog_ids]
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump({"reviewed_dialog_ids": ordered_reviewed}, file, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def next_unreviewed_index(dialog_ids: list[str], reviewed_dialog_ids: set[str], current_index: int) -> int | None:
    for offset in range(1, len(dialog_ids) + 1):
        candidate_index = (current_index + offset) % len(dialog_ids)
        if dialog_ids[candidate_index] not in reviewed_dialog_ids:
            return candidate_index

    return None


def set_current_index(index: int) -> None:
    st.session_state.current_index = index


def main() -> None:
    st.set_page_config(page_title="News Dialog Gold Annotation", layout="wide")
    st.title("Ручная gold-разметка новостных диалогов")
    try:
        dialog_ids = load_template_ids(TEMPLATE_PATH)
        dialogs = load_dialogs(DIALOGS_PATH)
        missing_dialog_ids = [dialog_id for dialog_id in dialog_ids if dialog_id not in dialogs]
        if missing_dialog_ids:
            raise ValueError(f"Dialogs missing from JSONL: {len(missing_dialog_ids)}")
        if not dialog_ids:
            raise ValueError("Annotation template does not contain dialog IDs.")
        allowed_dialog_ids = set(dialog_ids)
        annotations = load_annotations(GOLD_PATH, allowed_dialog_ids)
        reviewed_dialog_ids = load_reviewed(PROGRESS_PATH, allowed_dialog_ids)
    except (FileNotFoundError, ValueError, json.JSONDecodeError, KeyError) as error:
        st.error(str(error))
        st.stop()

    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    st.session_state.current_index = min(max(st.session_state.current_index, 0), len(dialog_ids) - 1)
    current_index = st.session_state.current_index
    current_dialog_id = dialog_ids[current_index]
    current_annotations = [row for row in annotations if row["dialog_id"] == current_dialog_id]

    reviewed_count = len(reviewed_dialog_ids)
    statistics_columns = st.columns(3)
    statistics_columns[0].metric("Reviewed", f"{reviewed_count} / {len(dialog_ids)}")
    statistics_columns[1].metric("Remaining", len(dialog_ids) - reviewed_count)
    statistics_columns[2].metric("Total annotations", len(annotations))
    st.progress(reviewed_count / len(dialog_ids))

    st.subheader(f"{current_index + 1} / {len(dialog_ids)}")
    st.caption(f"dialog_id: {current_dialog_id}")
    status = "Reviewed" if current_dialog_id in reviewed_dialog_ids else "Not reviewed"
    st.write(f"Status: {status} · Annotations: {len(current_annotations)}")
    st.text_area("Dialog text", dialogs[current_dialog_id], height=420, disabled=True)

    with st.form("add_annotation", clear_on_submit=True):
        form_columns = st.columns([1, 3])
        label = form_columns[0].selectbox("Label", LABELS)
        value = form_columns[1].text_input("Value")
        add_annotation = st.form_submit_button("Add annotation", type="primary")
    if add_annotation:
        normalized_value = value.strip()
        annotation = {"dialog_id": current_dialog_id, "label": label, "value": normalized_value}
        if not normalized_value:
            st.error("Value cannot be empty.")
        elif annotation in annotations:
            st.info("This annotation already exists.")
        else:
            annotations.append(annotation)
            save_annotations(GOLD_PATH, annotations)
            st.rerun()

    st.subheader("Annotations for this dialog")
    if current_annotations:
        st.dataframe(pd.DataFrame(current_annotations)[["label", "value"]], hide_index=True, use_container_width=True)
        for annotation_index, annotation in enumerate(current_annotations):
            delete_columns = st.columns([5, 1])
            delete_columns[0].write(f"{annotation['label']}: {annotation['value']}")
            if delete_columns[1].button("Delete", key=f"delete-{current_dialog_id}-{annotation_index}"):
                annotations.remove(annotation)
                save_annotations(GOLD_PATH, annotations)
                st.rerun()
    else:
        st.info("No annotations added. Mark the dialog as reviewed if it contains no target entities.")

    review_columns = st.columns(2)
    if current_dialog_id not in reviewed_dialog_ids:
        if review_columns[0].button("Mark dialog as reviewed", type="primary"):
            reviewed_dialog_ids.add(current_dialog_id)
            save_annotations(GOLD_PATH, annotations)
            save_reviewed(PROGRESS_PATH, reviewed_dialog_ids, dialog_ids)
            st.rerun()
    elif review_columns[0].button("Mark dialog as not reviewed"):
        reviewed_dialog_ids.remove(current_dialog_id)
        save_annotations(GOLD_PATH, annotations)
        save_reviewed(PROGRESS_PATH, reviewed_dialog_ids, dialog_ids)
        st.rerun()

    if review_columns[1].button("Next unreviewed"):
        target_index = next_unreviewed_index(dialog_ids, reviewed_dialog_ids, current_index)
        if target_index is None:
            st.success("All dialogs are reviewed.")
        else:
            set_current_index(target_index)
            st.rerun()

    navigation_columns = st.columns([1, 1, 2, 1])
    if navigation_columns[0].button("Previous", disabled=current_index == 0):
        set_current_index(current_index - 1)
        st.rerun()
    if navigation_columns[1].button("Next", disabled=current_index == len(dialog_ids) - 1):
        set_current_index(current_index + 1)
        st.rerun()
    target_number = navigation_columns[2].number_input("Dialog number", min_value=1, max_value=len(dialog_ids), value=current_index + 1)
    if navigation_columns[3].button("Go"):
        set_current_index(int(target_number) - 1)
        st.rerun()


if __name__ == "__main__":
    main()

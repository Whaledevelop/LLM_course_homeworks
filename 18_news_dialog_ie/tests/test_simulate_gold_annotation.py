import csv
import json

import pytest

from annotation_app import load_annotations, load_reviewed
from evaluation import annotation_template_fingerprint
from simulate_gold_annotation import EXPECTED_DIALOG_IDS, build_annotations, simulate_annotation


def test_build_annotations_uses_exact_text_spans() -> None:
    dialogs = {dialog_id: "" for dialog_id in EXPECTED_DIALOG_IDS}
    dialogs[EXPECTED_DIALOG_IDS[0]] = "Amazon was cited by Reuters."

    with pytest.raises(ValueError, match="Span not found"):
        build_annotations(EXPECTED_DIALOG_IDS, dialogs)


def test_simulation_writes_ui_compatible_gold_and_reviewed_state(tmp_path) -> None:
    template_path = tmp_path / "gold_annotation_template.csv"
    dialogs_path = tmp_path / "news_dialogs.jsonl"
    with template_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["dialog_id", "label", "value"])
        writer.writeheader()
        for dialog_id in EXPECTED_DIALOG_IDS:
            writer.writerow({"dialog_id": dialog_id})
    from simulate_gold_annotation import ANNOTATION_SPECS

    with dialogs_path.open("w", encoding="utf-8") as file:
        for dialog_id in EXPECTED_DIALOG_IDS:
            values = [value for _, value in ANNOTATION_SPECS[dialog_id]]
            file.write(json.dumps({"dialog_id": dialog_id, "text": " | ".join(values)}) + "\n")

    annotation_count, reviewed_count = simulate_annotation(tmp_path, create_backup=False)

    allowed_dialog_ids = set(EXPECTED_DIALOG_IDS)
    annotations = load_annotations(tmp_path / "gold_annotations.csv", allowed_dialog_ids)
    fingerprint = annotation_template_fingerprint(EXPECTED_DIALOG_IDS)
    reviewed = load_reviewed(tmp_path / "annotation_progress.json", allowed_dialog_ids, fingerprint)
    assert annotation_count == len(annotations)
    assert reviewed_count == 10
    assert reviewed == allowed_dialog_ids
    assert set(annotations[0]) == {"dialog_id", "label", "value"}
    assert not any(row["dialog_id"] == EXPECTED_DIALOG_IDS[-1] for row in annotations)


def test_simulation_rejects_a_different_template(tmp_path) -> None:
    template_path = tmp_path / "gold_annotation_template.csv"
    template_path.write_text("dialog_id,label,value\nother,,\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        simulate_annotation(tmp_path, create_backup=False)

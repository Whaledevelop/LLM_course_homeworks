import csv
import json

from annotation_app import load_annotations, load_reviewed, next_unreviewed_index, save_annotations, save_reviewed


def test_annotations_round_trip_filters_empty_and_duplicate_rows(tmp_path) -> None:
    path = tmp_path / "gold.csv"
    annotations = [
        {"dialog_id": "one", "label": "ORG", "value": "OpenAI"},
        {"dialog_id": "one", "label": "ORG", "value": "OpenAI"},
        {"dialog_id": "two", "label": "LOC", "value": ""},
    ]
    save_annotations(path, annotations)
    loaded = load_annotations(path, {"one", "two"})

    assert loaded == [{"dialog_id": "one", "label": "ORG", "value": "OpenAI"}]
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2


def test_reviewed_round_trip_preserves_template_order(tmp_path) -> None:
    path = tmp_path / "progress.json"
    save_reviewed(path, {"three", "one"}, ["one", "two", "three"])
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload == {"reviewed_dialog_ids": ["one", "three"]}
    assert load_reviewed(path, {"one", "two", "three"}) == {"one", "three"}


def test_next_unreviewed_wraps_and_reports_completion() -> None:
    dialog_ids = ["one", "two", "three"]

    assert next_unreviewed_index(dialog_ids, {"one", "two"}, 1) == 2
    assert next_unreviewed_index(dialog_ids, {"two", "three"}, 2) == 0
    assert next_unreviewed_index(dialog_ids, set(dialog_ids), 0) is None

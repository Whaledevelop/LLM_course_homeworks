import csv
import json

from annotation_app import load_annotations, load_reviewed, next_unreviewed_index, save_annotations, save_reviewed
from evaluation import annotation_template_fingerprint


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
    dialog_ids = ["one", "two", "three"]
    fingerprint = annotation_template_fingerprint(dialog_ids)
    save_reviewed(path, {"three", "one"}, dialog_ids, fingerprint)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload == {"template_fingerprint": fingerprint, "reviewed_dialog_ids": ["one", "three"]}
    assert load_reviewed(path, set(dialog_ids), fingerprint) == {"one", "three"}


def test_reviewed_progress_resets_for_different_template(tmp_path) -> None:
    path = tmp_path / "progress.json"
    first_ids = ["one", "two"]
    save_reviewed(path, {"one"}, first_ids, annotation_template_fingerprint(first_ids))

    assert load_reviewed(path, {"one", "three"}, annotation_template_fingerprint(["one", "three"])) == set()


def test_next_unreviewed_wraps_and_reports_completion() -> None:
    dialog_ids = ["one", "two", "three"]

    assert next_unreviewed_index(dialog_ids, {"one", "two"}, 1) == 2
    assert next_unreviewed_index(dialog_ids, {"two", "three"}, 2) == 0
    assert next_unreviewed_index(dialog_ids, set(dialog_ids), 0) is None

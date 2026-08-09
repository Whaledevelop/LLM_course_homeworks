import csv
import json

import pytest

from evaluation import annotation_template_fingerprint
from annotation_workspace import prepare_annotation_workspace, validate_gold
from run_demo import MODEL_ALIASES, build_parser, clear_cache, clear_classifier_cache, clear_dataset_cache, clear_dataset_outputs, parse_profiles, validate_smoke_dataset


def test_default_gold_size_is_ten() -> None:
    assert build_parser().parse_args([]).gold_size == 10


def test_default_extraction_output_limit_is_1024_tokens() -> None:
    assert build_parser().parse_args([]).max_new_tokens == 1024


def test_default_profiles_use_local_models_below_three_billion_parameters() -> None:
    assert build_parser().parse_args([]).profiles == ["qwen-fp16", "qwen-int8", "gemma-fp16", "gemma-int8"]
    assert MODEL_ALIASES == {
        "qwen": "Qwen/Qwen3-1.7B",
        "gemma": "google/gemma-2-2b-it",
    }
    assert parse_profiles(["qwen-fp16", "gemma-int8"]) == ["qwen-fp16", "gemma-int8"]


def test_benchmark_limit_must_be_positive() -> None:
    assert build_parser().parse_args(["--benchmark-limit", "5"]).benchmark_limit == 5


def test_quality_debug_is_disabled_by_default_and_can_be_enabled() -> None:
    assert build_parser().parse_args([]).quality_debug is False
    assert build_parser().parse_args(["--quality-debug"]).quality_debug is True


def test_smoke_mode_requires_existing_dataset(tmp_path) -> None:
    dataset_path = tmp_path / "news_dialogs.jsonl"

    with pytest.raises(ValueError):
        validate_smoke_dataset(5, False, dataset_path)
    dataset_path.write_text("data", encoding="utf-8")
    validate_smoke_dataset(5, False, dataset_path)


def test_default_annotation_template_contains_ten_dialogs(tmp_path) -> None:
    dialog_ids = [f"dialog-{index}" for index in range(build_parser().parse_args([]).gold_size)]

    prepare_annotation_workspace(tmp_path, dialog_ids)

    template_path = tmp_path / "gold_annotation_template.csv"
    with template_path.open("r", encoding="utf-8", newline="") as file:
        assert len(list(csv.DictReader(file))) == 10


def test_prepare_annotation_workspace_resets_changed_template_with_backup(tmp_path) -> None:
    template_path = tmp_path / "gold_annotation_template.csv"
    gold_path = tmp_path / "gold_annotations.csv"
    progress_path = tmp_path / "annotation_progress.json"
    template_path.write_text("dialog_id,label,value\nold,,\n", encoding="utf-8")
    gold_path.write_text("dialog_id,label,value\nold,ORG,Old Org\n", encoding="utf-8")
    progress_path.write_text('{"reviewed_dialog_ids":["old"]}', encoding="utf-8")

    prepare_annotation_workspace(tmp_path, ["one", "two"])

    with template_path.open("r", encoding="utf-8", newline="") as file:
        assert [row["dialog_id"] for row in csv.DictReader(file)] == ["one", "two"]
    assert len(list((tmp_path / "annotation_backups").iterdir())) == 1
    assert list(csv.DictReader(gold_path.open("r", encoding="utf-8", newline=""))) == []
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["reviewed_dialog_ids"] == []


def test_prepare_annotation_workspace_preserves_matching_progress(tmp_path) -> None:
    prepare_annotation_workspace(tmp_path, ["one", "two"])
    progress_path = tmp_path / "annotation_progress.json"
    progress_path.write_text('{"reviewed_dialog_ids":["one"]}', encoding="utf-8")

    prepare_annotation_workspace(tmp_path, ["one", "two"])

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["reviewed_dialog_ids"] == ["one"]
    assert progress["template_fingerprint"] == annotation_template_fingerprint(["one", "two"])


def test_validate_gold_accepts_reviewed_dialog_without_annotations(tmp_path) -> None:
    prepare_annotation_workspace(tmp_path, ["one", "two"])
    progress_path = tmp_path / "annotation_progress.json"
    progress_path.write_text(
        json.dumps(
            {
                "template_fingerprint": annotation_template_fingerprint(["one", "two"]),
                "reviewed_dialog_ids": ["one", "two"],
            }
        ),
        encoding="utf-8",
    )

    validate_gold(
        tmp_path / "gold_annotations.csv",
        tmp_path / "gold_annotation_template.csv",
        progress_path,
        2,
        False,
    )


def test_clear_dataset_cache_removes_only_dataset_file(tmp_path) -> None:
    dataset_path = tmp_path / "news_dialogs.jsonl"
    other_path = tmp_path / "benchmark_results.csv"
    dataset_path.write_text("data", encoding="utf-8")
    other_path.write_text("metrics", encoding="utf-8")

    clear_dataset_cache(dataset_path)

    assert not dataset_path.exists()
    assert other_path.exists()


def test_clear_dataset_outputs_preserves_dataset_and_annotations(tmp_path) -> None:
    benchmark_path = tmp_path / "benchmark_results.csv"
    dataset_path = tmp_path / "news_dialogs.jsonl"
    gold_path = tmp_path / "gold_annotations.csv"
    benchmark_path.write_text("metrics", encoding="utf-8")
    dataset_path.write_text("dialogs", encoding="utf-8")
    gold_path.write_text("gold", encoding="utf-8")

    clear_dataset_outputs(tmp_path)

    assert not benchmark_path.exists()
    assert dataset_path.exists()
    assert gold_path.exists()


def test_cache_reset_preserves_classifier_cache(tmp_path) -> None:
    classifier_path = tmp_path / "news_classifier.jsonl"
    extraction_path = tmp_path / "extractor.jsonl"
    classifier_path.write_text("classifier", encoding="utf-8")
    extraction_path.write_text("extraction", encoding="utf-8")

    clear_cache(tmp_path)

    assert classifier_path.exists()
    assert not extraction_path.exists()
    clear_classifier_cache(classifier_path)
    assert not classifier_path.exists()

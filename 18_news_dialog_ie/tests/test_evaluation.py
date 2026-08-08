import csv

import pytest

from evaluation import evaluate, normalize
from schemas import ExtractedItem, ExtractionResult


def test_normalize_collapses_case_and_spaces() -> None:
    assert normalize("  New   York ") == "new york"


def test_evaluation_calculates_micro_macro_and_errors(tmp_path) -> None:
    gold_path = tmp_path / "gold.csv"
    with gold_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["dialog_id", "label", "value"])
        writer.writeheader()
        writer.writerows([
            {"dialog_id": "1", "label": "ORG", "value": "OpenAI"},
            {"dialog_id": "1", "label": "LOC", "value": "London"},
        ])
    results = [ExtractionResult("1", "test", entities=[ExtractedItem("ORG", "OpenAI"), ExtractedItem("PERSON", "London")])]
    report = evaluate(results, gold_path)

    assert report.precision == pytest.approx(0.5)
    assert report.recall == pytest.approx(0.5)
    assert report.micro_f1 == pytest.approx(0.5)
    assert len(report.errors) == 2

from pathlib import Path

import pytest
from pydantic import ValidationError

from tutor_bot.evaluation.golden_loader import load_golden_dataset


def test_loads_json_with_utf8_bom(tmp_path: Path) -> None:
    golden_path = tmp_path / "goldens.json"
    golden_path.write_text(
        """
        {
          "version": 1,
          "retrieval_cases": [
            {
              "id": "missing-information",
              "question": "Как настроить Kubernetes?",
              "expected_note_ids": [],
              "should_find_answer": false
            }
          ]
        }
        """,
        encoding="utf-8-sig",
    )

    dataset = load_golden_dataset(golden_path)

    assert dataset.version == 1
    assert len(dataset.retrieval_cases) == 1


def test_rejects_invalid_dataset(tmp_path: Path) -> None:
    golden_path = tmp_path / "goldens.json"
    golden_path.write_text(
        """
        {
          "version": 1,
          "retrieval_cases": [
            {
              "id": "invalid",
              "question": "Что такое GC?",
              "expected_note_ids": [],
              "should_find_answer": true
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_golden_dataset(golden_path)

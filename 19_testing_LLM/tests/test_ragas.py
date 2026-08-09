import os
from typing import Any

import pytest

from app.config import Settings
from tests.evaluate import (
    RagasEvaluator,
    load_goldens,
    load_report,
    save_report,
)


FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD") or "0.70")

THRESHOLDS = {
    "faithfulness": FAITHFULNESS_THRESHOLD,
    "answer_relevancy": float(os.getenv("ANSWER_RELEVANCY_THRESHOLD", "0.70")),
    "context_recall": float(os.getenv("CONTEXT_RECALL_THRESHOLD", "0.70")),
}


@pytest.fixture(scope="session")
def ragas_report() -> dict[str, Any]:
    if os.getenv("RAGAS_USE_EXISTING_REPORT") == "1":
        return load_report()

    evaluator = RagasEvaluator(Settings.from_env())
    report = evaluator.evaluate(load_goldens())
    save_report(report)

    return report


@pytest.mark.parametrize("metric_name", THRESHOLDS)
def test_ragas_quality_gate(
    ragas_report: dict[str, Any],
    metric_name: str,
) -> None:
    score = ragas_report["averages"][metric_name]
    threshold = THRESHOLDS[metric_name]

    assert score >= threshold, (
        f"{metric_name} ниже порога: {score:.3f} < {threshold:.3f}"
    )

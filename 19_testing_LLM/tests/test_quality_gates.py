from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from dataset import load_corpus
from evaluator import THRESHOLDS, load_goldens, run_evaluation
from rag_app import LocalRagApplication


def test_ragas_quality_gates() -> None:
    documents = load_corpus(PROJECT_DIR / "data" / "corpus.jsonl")
    goldens = load_goldens(PROJECT_DIR / "tests" / "goldens.json")
    app = LocalRagApplication(documents, top_k=2)

    summary, rows = run_evaluation(app, goldens)

    assert len(rows) >= 10
    assert summary.faithfulness >= THRESHOLDS["faithfulness"]
    assert summary.answer_relevance >= THRESHOLDS["answer_relevance"]
    assert summary.context_recall >= THRESHOLDS["context_recall"]

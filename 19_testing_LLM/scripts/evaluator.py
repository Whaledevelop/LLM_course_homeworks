from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rag_app import LocalRagApplication, tokenize


@dataclass(frozen=True)
class GoldenExample:
    id: str
    question: str
    reference: str
    expected_context_ids: list[str]
    key_facts: list[str]


@dataclass(frozen=True)
class EvaluationRow:
    id: str
    question: str
    answer: str
    contexts: list[str]
    context_ids: list[str]
    reference: str
    faithfulness: float
    answer_relevance: float
    context_recall: float
    passed: bool


@dataclass(frozen=True)
class EvaluationSummary:
    examples: int
    faithfulness: float
    answer_relevance: float
    context_recall: float
    pass_rate: float
    thresholds: dict[str, float]


THRESHOLDS = {
    "faithfulness": 0.70,
    "answer_relevance": 0.65,
    "context_recall": 0.75,
}


def load_goldens(path: Path) -> list[GoldenExample]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    goldens = []
    for item in payload:
        goldens.append(GoldenExample(**item))

    return goldens


def run_evaluation(app: LocalRagApplication, goldens: list[GoldenExample]) -> tuple[EvaluationSummary, list[EvaluationRow]]:
    rows = []
    for golden in goldens:
        result = app.answer(golden.question)
        faithfulness = score_faithfulness(result.answer, result.contexts, golden.key_facts)
        answer_relevance = score_answer_relevance(golden.question, result.answer, golden.reference)
        context_recall = score_context_recall(golden.expected_context_ids, result.context_ids)
        passed = (
            faithfulness >= THRESHOLDS["faithfulness"]
            and answer_relevance >= THRESHOLDS["answer_relevance"]
            and context_recall >= THRESHOLDS["context_recall"]
        )
        rows.append(
            EvaluationRow(
                id=golden.id,
                question=golden.question,
                answer=result.answer,
                contexts=result.contexts,
                context_ids=result.context_ids,
                reference=golden.reference,
                faithfulness=faithfulness,
                answer_relevance=answer_relevance,
                context_recall=context_recall,
                passed=passed,
            )
        )

    summary = EvaluationSummary(
        examples=len(rows),
        faithfulness=mean([row.faithfulness for row in rows]),
        answer_relevance=mean([row.answer_relevance for row in rows]),
        context_recall=mean([row.context_recall for row in rows]),
        pass_rate=mean([1.0 if row.passed else 0.0 for row in rows]),
        thresholds=THRESHOLDS,
    )

    return summary, rows


def score_faithfulness(answer: str, contexts: list[str], key_facts: list[str]) -> float:
    context_text = " ".join(contexts).lower()
    supported_facts = 0
    for fact in key_facts:
        fact_tokens = tokenize(fact)
        context_tokens = tokenize(context_text)
        if fact_tokens and len(fact_tokens.intersection(context_tokens)) / len(fact_tokens) >= 0.75:
            supported_facts += 1

    answer_tokens = tokenize(answer)
    context_tokens = tokenize(context_text)
    unsupported_answer_tokens = answer_tokens.difference(context_tokens)
    support_penalty = min(len(unsupported_answer_tokens) / max(len(answer_tokens), 1), 1.0)
    fact_score = supported_facts / max(len(key_facts), 1)

    return round(max(fact_score - support_penalty * 0.25, 0.0), 3)


def score_answer_relevance(question: str, answer: str, reference: str) -> float:
    expected_tokens = tokenize(reference)
    answer_tokens = tokenize(answer)
    overlap = expected_tokens.intersection(answer_tokens)

    return round(len(overlap) / max(len(expected_tokens), 1), 3)


def score_context_recall(expected_context_ids: list[str], actual_context_ids: list[str]) -> float:
    expected_ids = set(expected_context_ids)
    actual_ids = set(actual_context_ids)
    recalled_ids = expected_ids.intersection(actual_ids)

    return round(len(recalled_ids) / max(len(expected_ids), 1), 3)


def write_json_report(path: Path, summary: EvaluationSummary, rows: list[EvaluationRow]) -> None:
    payload = {
        "summary": asdict(summary),
        "rows": [asdict(row) for row in rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_html_report(path: Path, summary: EvaluationSummary, rows: list[EvaluationRow]) -> None:
    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(row.id)}</td>"
            f"<td>{row.faithfulness:.3f}</td>"
            f"<td>{row.answer_relevance:.3f}</td>"
            f"<td>{row.context_recall:.3f}</td>"
            f"<td>{'pass' if row.passed else 'fail'}</td>"
            "</tr>"
        )

    content = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>LLM quality report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; }}
    th {{ background: #f2f2f2; text-align: left; }}
  </style>
</head>
<body>
  <h1>LLM quality report</h1>
  <p>Faithfulness: {summary.faithfulness:.3f}</p>
  <p>Answer relevance: {summary.answer_relevance:.3f}</p>
  <p>Context recall: {summary.context_recall:.3f}</p>
  <p>Pass rate: {summary.pass_rate:.3f}</p>
  <table>
    <thead>
      <tr><th>ID</th><th>Faithfulness</th><th>Answer relevance</th><th>Context recall</th><th>Status</th></tr>
    </thead>
    <tbody>
      {''.join(table_rows)}
    </tbody>
  </table>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write(content)


def mean(values: list[float]) -> float:
    if not values:
        return 0.0

    return round(sum(values) / len(values), 3)

import csv
from dataclasses import dataclass
from pathlib import Path

from schemas import ExtractionResult


ALLOWED_LABELS = {"PERSON", "ORG", "LOC", "EVENT", "DATE", "IMPACT", "SOURCE"}


@dataclass(frozen=True)
class EvaluationReport:
    precision: float
    recall: float
    micro_f1: float
    macro_f1: float
    per_class: list[dict]
    errors: list[dict]
    evaluated_dialogs: int


def evaluate(results: list[ExtractionResult], gold_path: Path) -> EvaluationReport:
    gold_labels = load_gold_labels(gold_path)
    expected = set()
    predicted = set()
    result_ids = {result.dialog_id for result in results}
    evaluated_ids = set(gold_labels) & result_ids
    for dialog_id in evaluated_ids:
        items = gold_labels[dialog_id]
        expected.update((dialog_id, label, normalize(value)) for label, value in items)
    for result in results:
        if result.dialog_id not in evaluated_ids:
            continue
        items = list(result.entities) + list(result.events)
        predicted.update(
            (result.dialog_id, item.label, normalize(item.value))
            for item in items
            if item.label in ALLOWED_LABELS and item.value.strip()
        )

    true_positive = len(predicted & expected)
    precision = safe_divide(true_positive, len(predicted))
    recall = safe_divide(true_positive, len(expected))
    micro_f1 = f1_score(precision, recall)
    per_class = []
    for label in sorted(ALLOWED_LABELS):
        expected_class = {item for item in expected if item[1] == label}
        predicted_class = {item for item in predicted if item[1] == label}
        class_true_positive = len(expected_class & predicted_class)
        class_precision = safe_divide(class_true_positive, len(predicted_class))
        class_recall = safe_divide(class_true_positive, len(expected_class))
        per_class.append(
            {
                "label": label,
                "precision": class_precision,
                "recall": class_recall,
                "f1": f1_score(class_precision, class_recall),
                "support": len(expected_class),
            }
        )
    supported_scores = [row["f1"] for row in per_class if row["support"]]
    macro_f1 = sum(supported_scores) / len(supported_scores) if supported_scores else 0.0
    errors = [build_error_row(item, "false_negative") for item in sorted(expected - predicted)]
    errors.extend(build_error_row(item, "false_positive") for item in sorted(predicted - expected))

    return EvaluationReport(precision, recall, micro_f1, macro_f1, per_class, errors, len(evaluated_ids))


def load_gold_labels(gold_path: Path) -> dict[str, set[tuple[str, str]]]:
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold annotations not found: {gold_path}")
    labels: dict[str, set[tuple[str, str]]] = {}
    with gold_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            dialog_id = row["dialog_id"].strip()
            label = row["label"].strip().upper()
            value = row["value"].strip()
            if label not in ALLOWED_LABELS:
                raise ValueError(f"Unsupported gold label: {label}")
            if dialog_id and value:
                labels.setdefault(dialog_id, set()).add((label, value))

    return labels


def write_annotation_template(path: Path, dialog_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["dialog_id", "label", "value"])
        writer.writeheader()
        for dialog_id in dialog_ids:
            writer.writerow({"dialog_id": dialog_id, "label": "", "value": ""})


def write_evaluation_csvs(per_class_path: Path, errors_path: Path, extractor: str, batch_size: int, report: EvaluationReport) -> None:
    per_class_path.parent.mkdir(parents=True, exist_ok=True)
    append_rows(per_class_path, [dict(extractor=extractor, batch_size=batch_size, **row) for row in report.per_class])
    append_rows(errors_path, [dict(extractor=extractor, batch_size=batch_size, **row) for row in report.errors])


def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def normalize(value: str) -> str:
    return " ".join(value.lower().strip().split())


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def f1_score(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def build_error_row(item: tuple[str, str, str], error_type: str) -> dict:
    dialog_id, label, value = item

    return {"dialog_id": dialog_id, "error_type": error_type, "label": label, "value": value}

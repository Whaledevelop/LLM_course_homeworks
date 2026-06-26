import csv
from pathlib import Path

from schemas import ExtractedItem, ExtractionResult


GOLD_LABELS = {
    "sample-001": {
        ("PERSON", "Joe Biden"),
        ("PERSON", "Jens Stoltenberg"),
        ("ORG", "Reuters"),
        ("ORG", "NATO"),
        ("LOC", "Washington"),
        ("DATE", "April 4, 2024"),
        ("EVENT", "meeting"),
        ("IMPACT", "strengthened NATO coordination"),
        ("SOURCE", "Reuters"),
    },
    "sample-002": {
        ("ORG", "BBC"),
        ("ORG", "Apple"),
        ("LOC", "California"),
        ("DATE", "May 2, 2024"),
        ("EVENT", "share buyback"),
        ("EVENT", "quarterly earnings"),
        ("IMPACT", "boosted investor confidence"),
        ("SOURCE", "BBC"),
    },
    "sample-003": {
        ("ORG", "AP News"),
        ("LOC", "Hualien"),
        ("LOC", "Taiwan"),
        ("DATE", "April 3, 2024"),
        ("EVENT", "earthquake"),
        ("IMPACT", "damaging buildings"),
        ("IMPACT", "disrupting transport"),
        ("SOURCE", "AP News"),
    },
    "sample-004": {
        ("ORG", "The Guardian"),
        ("ORG", "Tesla"),
        ("LOC", "Texas"),
        ("LOC", "California"),
        ("DATE", "2024"),
        ("EVENT", "layoffs"),
        ("IMPACT", "affect production planning"),
        ("IMPACT", "signal cost pressure"),
        ("SOURCE", "The Guardian"),
    },
    "sample-005": {
        ("ORG", "CNN"),
        ("ORG", "European Union"),
        ("ORG", "Microsoft"),
        ("LOC", "Brussels"),
        ("DATE", "June 25, 2024"),
        ("EVENT", "investigation"),
        ("IMPACT", "harmed competition"),
        ("SOURCE", "CNN"),
    },
}


def evaluate(results: list[ExtractionResult], gold_path: Path | None = None) -> tuple[float, float, float]:
    gold_labels = load_gold_labels(gold_path)
    predicted = set()
    expected = set()
    for result in results:
        gold_items = resolve_gold_items(result.dialog_id, gold_labels)
        if not gold_items:
            continue
        expected.update((result.dialog_id, label, normalize(value)) for label, value in gold_items)
        items = list(result.entities) + list(result.events)
        predicted.update((result.dialog_id, item.label, normalize(item.value)) for item in items)

    true_positive = len(predicted & expected)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return precision, recall, f1


def load_gold_labels(gold_path: Path | None = None) -> dict[str, set[tuple[str, str]]]:
    if not gold_path or not gold_path.exists():
        return GOLD_LABELS

    labels: dict[str, set[tuple[str, str]]] = {}
    with gold_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            dialog_id = row["dialog_id"]
            label = row["label"]
            value = row["value"]
            labels.setdefault(dialog_id, set()).add((label, value))

    return labels


def resolve_gold_items(dialog_id: str, gold_labels: dict[str, set[tuple[str, str]]]) -> set[tuple[str, str]]:
    if dialog_id in gold_labels:
        return gold_labels[dialog_id]
    if not dialog_id.startswith("sample-"):
        return set()

    try:
        sample_number = int(dialog_id.split("-", maxsplit=1)[1])
    except ValueError:
        return set()

    base_number = ((sample_number - 1) % 5) + 1
    base_dialog_id = f"sample-{base_number:03d}"

    return gold_labels.get(base_dialog_id, set())


def write_gold_csv(path: Path, dialog_ids: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_dialog_ids = dialog_ids or sorted(GOLD_LABELS)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["dialog_id", "label", "value"])
        writer.writeheader()
        for dialog_id in selected_dialog_ids:
            items = resolve_gold_items(dialog_id, GOLD_LABELS)
            for label, value in sorted(items):
                writer.writerow({"dialog_id": dialog_id, "label": label, "value": value})


def normalize(value: str) -> str:
    return " ".join(value.lower().strip().split())

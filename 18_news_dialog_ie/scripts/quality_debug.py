import json

from evaluation import ALLOWED_LABELS, load_gold_labels, normalize


def print_quality_debug(profile: str, batch_size: int, results, gold_path, evaluation_dialog_ids: set[str]) -> None:
    gold_labels = load_gold_labels(gold_path)
    evaluated_results = [result for result in results if result.dialog_id in evaluation_dialog_ids]
    print(f"[{profile}] Quality debug | batch={batch_size} | evaluated dialogs={len(evaluated_results)}", flush=True)
    if not evaluated_results:
        print(f"[{profile}] No gold dialogs are present in this benchmark subset.", flush=True)

        return
    for result in evaluated_results:
        gold = {
            (label, normalize(value))
            for label, value in gold_labels.get(result.dialog_id, set())
        }
        predicted_items = list(result.entities) + list(result.events)
        predicted = {
            (item.label, normalize(item.value))
            for item in predicted_items
            if item.label in ALLOWED_LABELS and item.value.strip()
        }
        print(f"[{profile}] Dialog: {result.dialog_id}", flush=True)
        print(f"[{profile}] Parse valid: {result.parse_valid} | error: {result.error or 'none'}", flush=True)
        print(f"[{profile}] Raw response: {result.raw_response!r}", flush=True)
        print(f"[{profile}] Parsed entities: {serialize_items(result.entities)}", flush=True)
        print(f"[{profile}] Parsed events: {serialize_items(result.events)}", flush=True)
        print(f"[{profile}] Parsed relations: {json.dumps(result.relations, ensure_ascii=False)}", flush=True)
        print(f"[{profile}] Gold normalized: {serialize_pairs(gold)}", flush=True)
        print(f"[{profile}] Predicted normalized: {serialize_pairs(predicted)}", flush=True)
        print(f"[{profile}] True positives: {serialize_pairs(gold & predicted)}", flush=True)
        print(f"[{profile}] False positives: {serialize_pairs(predicted - gold)}", flush=True)
        print(f"[{profile}] False negatives: {serialize_pairs(gold - predicted)}", flush=True)


def serialize_items(items) -> str:
    rows = [{"label": item.label, "value": item.value} for item in items]

    return json.dumps(rows, ensure_ascii=False)


def serialize_pairs(items: set[tuple[str, str]]) -> str:
    rows = [{"label": label, "value": value} for label, value in sorted(items)]

    return json.dumps(rows, ensure_ascii=False)

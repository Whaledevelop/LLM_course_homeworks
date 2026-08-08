import re
from pathlib import Path


def print_dataset_progress(event: str, payload: dict, gold_size: int) -> None:
    if event == "start":
        print(f"[Dataset] Target: {payload['target']} NEWS dialogs")
        print(f"[Dataset] Gold subset: {gold_size} dialogs")
        print(f"[Classifier] {payload['model']}")
        print(f"[Classifier] Device: {payload['device']}")
        print(f"[Classifier] Batch size: {payload['batch_size']}")
        print(f"[Classifier] Max input tokens: {payload['max_input_tokens']}")

        return
    if event == "classification":
        cache_suffix = " [CACHE]" if payload["cache_hit"] else ""
        print(
            f"[{payload['index']:04d}] Stage 1: PASS | LLM: {payload['classification']}{cache_suffix} | "
            f"NEWS collected: {payload['news_collected']}/{payload['target']}"
        )
        print(f"       {normalize_preview(payload['text'])}")
        if payload["classification"] == "INVALID":
            print(f'       Raw classifier output: "{normalize_preview(payload["raw_output"], 160)}"')

        return
    if event == "scan":
        print(
            f"[Scan] rows={payload['rows_seen']} | stage1={payload['stage1_passed']} | "
            f"classified={payload['llm_classified']} | news={payload['llm_news']}/{payload['target']}"
        )


def print_dataset_summary(stats: dict, dataset_path: Path, template_path: Path, gold_size: int, news_count: int) -> None:
    print("Dataset preparation complete")
    print()
    print(f"Rows scanned: {stats['rows_seen']}")
    print(f"Stage 1 passed: {stats['stage1_passed']}")
    print(f"LLM classified: {stats['llm_classified']}")
    print(f"NEWS: {news_count}")
    print(f"NOT_NEWS: {stats['llm_not_news']}")
    print(f"INVALID: {stats['llm_invalid']}")
    print(f"Classifier cache: {stats['classifier_cache_hits']} hits")
    print(f"Dataset: {dataset_path}")
    print(f"Gold template: {template_path}")
    print(f"Gold dialogs: {gold_size}")


def normalize_preview(text: str, limit: int = 120) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized

    return normalized[: limit - 3].rstrip() + "..."

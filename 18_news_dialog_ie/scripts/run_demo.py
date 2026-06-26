import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from benchmark import ExtractionBenchmark, write_benchmark_csv
from dataset import load_news_dialogs
from evaluation import write_gold_csv
from extractors import RuleBasedNewsExtractor, SpacyNewsExtractor, TransformersJsonExtractor
from schemas import ExtractionResult


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--with-spacy", action="store_true")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--llm-quantized", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    data_dir = project_dir / "data"
    cache_dir = data_dir / "cache"
    if args.rebuild_cache:
        clear_cache(cache_dir)

    dialogs = load_news_dialogs(args.sample_size, args.seed, data_dir / "news_dialogs.jsonl")
    extractors = [RuleBasedNewsExtractor()]
    if args.with_spacy:
        extractors.append(SpacyNewsExtractor())
    if args.llm_model:
        extractors.append(TransformersJsonExtractor(args.llm_model, args.llm_quantized))

    benchmark = ExtractionBenchmark(cache_dir)
    benchmark_results = []
    all_extractions = []
    for extractor in extractors:
        result, extractions = benchmark.run(extractor, dialogs, args.batch_size)
        benchmark_results.append(result)
        all_extractions.extend(extractions)
        print_result(result)

    write_benchmark_csv(data_dir / "benchmark_results.csv", benchmark_results)
    write_gold_csv(data_dir / "gold_annotations.csv", [dialog.dialog_id for dialog in dialogs])
    write_predictions_csv(data_dir / "extraction_predictions.csv", all_extractions)
    write_json(data_dir / "extractions.json", [asdict(result) for result in all_extractions[:20]])


def clear_cache(cache_dir: Path) -> None:
    if not cache_dir.exists():
        return
    for path in cache_dir.glob("*.jsonl"):
        path.unlink()


def print_result(result) -> None:
    print(
        f"{result.extractor}: docs={result.examples}, batch={result.batch_size}, "
        f"docs/sec={result.docs_per_second:.2f}, latency={result.mean_latency_ms:.2f} ms, "
        f"rss={result.peak_rss_mb:.1f} MB, precision={result.precision:.3f}, "
        f"recall={result.recall:.3f}, f1={result.f1:.3f}"
    )


def write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)


def write_predictions_csv(path: Path, results: list[ExtractionResult]) -> None:
    fieldnames = ["dialog_id", "extractor", "kind", "label", "value", "start", "end", "confidence"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for item in result.entities:
                writer.writerow(build_prediction_row(result, "entity", item))
            for item in result.events:
                writer.writerow(build_prediction_row(result, "event", item))


def build_prediction_row(result: ExtractionResult, kind: str, item) -> dict:
    return {
        "dialog_id": result.dialog_id,
        "extractor": result.extractor,
        "kind": kind,
        "label": item.label,
        "value": item.value,
        "start": item.start,
        "end": item.end,
        "confidence": item.confidence,
    }


if __name__ == "__main__":
    main()

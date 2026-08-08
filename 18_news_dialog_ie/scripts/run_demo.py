import argparse
import csv
import gc
import json
import os
from collections import Counter
from dataclasses import asdict
from functools import partial
from pathlib import Path

from annotation_workspace import prepare_annotation_workspace, read_template_dialog_ids, validate_gold
from benchmark import ExtractionBenchmark, write_benchmark_csv
from dataset import load_news_dialogs
from dataset_progress import print_dataset_progress, print_dataset_summary
from evaluation import load_gold_labels, write_evaluation_csvs
from extractors import RuleBasedNewsExtractor, SpacyNewsExtractor, TransformersJsonExtractor
from news_classifier import DEFAULT_MAX_INPUT_TOKENS, DEFAULT_MODEL
from schemas import ExtractionResult


MODEL_ALIASES = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "openchat": "openchat/openchat-3.5-0106",
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    data_dir = project_dir / "data"
    cache_dir = data_dir / "cache"
    classifier_cache_path = cache_dir / "news_classifier.jsonl"
    dataset_path = data_dir / "news_dialogs.jsonl"
    dataset_stats_path = data_dir / "dataset_stats.json"
    template_path = data_dir / "gold_annotation_template.csv"
    progress_path = data_dir / "annotation_progress.json"
    gold_path = Path(args.gold_path) if args.gold_path else data_dir / "gold_annotations.csv"
    if args.rebuild_cache:
        clear_cache(cache_dir)
    if args.rebuild_classifier_cache:
        clear_classifier_cache(classifier_cache_path)
    if args.rebuild_dataset:
        clear_dataset_cache(dataset_path)
        clear_dataset_outputs(data_dir)

    dialogs, filtering_stats, classifier_info = load_news_dialogs(
        args.sample_size,
        args.seed,
        dataset_path,
        classifier_cache_path,
        args.news_classifier_model,
        args.news_classifier_batch_size,
        args.news_classifier_max_input_tokens,
        args.allow_synthetic,
        partial(print_dataset_progress, gold_size=args.gold_size),
        args.news_classifier_sanity_check,
    )
    validate_dataset(dialogs, args.sample_size, args.allow_synthetic)
    if filtering_stats is not None or not dataset_stats_path.exists():
        write_dataset_stats(dataset_stats_path, dialogs, filtering_stats, classifier_info)
    if args.prepare_annotations:
        prepare_annotation_workspace(data_dir, [dialog.dialog_id for dialog in dialogs[:args.gold_size]])
        if filtering_stats is not None:
            print_dataset_summary(
                filtering_stats,
                dataset_path,
                template_path,
                min(args.gold_size, len(dialogs)),
                len(dialogs),
            )
        else:
            print(f"Annotation template written for {min(args.gold_size, len(dialogs))} dialogs.")

        return
    validate_gold(gold_path, template_path, progress_path, args.gold_size, args.allow_incomplete_gold)
    write_gold_stats(data_dir / "gold_stats.json", gold_path)
    reset_output_files(data_dir)
    evaluation_dialog_ids = set(load_gold_labels(gold_path)) if args.allow_incomplete_gold else set(read_template_dialog_ids(template_path))
    benchmark = ExtractionBenchmark(cache_dir, gold_path, evaluation_dialog_ids)
    benchmark_results = []
    all_extractions = []
    for profile in parse_profiles(args.profiles):
        extractor = create_extractor(profile, args.max_new_tokens)
        successful_batch = None
        selected_extractions = []
        for batch_size in args.batch_sizes:
            try:
                result, extractions, report = benchmark.run(extractor, dialogs, batch_size)
            except RuntimeError as error:
                if "out of memory" not in str(error).lower():
                    raise
                clear_gpu_memory()
                print(f"{extractor.name}: batch={batch_size} skipped after OOM")
                continue
            benchmark_results.append(result)
            selected_extractions = extractions
            write_evaluation_csvs(data_dir / "per_class_metrics.csv", data_dir / "extraction_errors.csv", result.extractor, batch_size, report)
            successful_batch = batch_size
            print_result(result)
        if successful_batch is not None:
            print(f"{extractor.name}: largest successful batch={successful_batch}")
            all_extractions.extend(selected_extractions)
        del extractor
        clear_gpu_memory()

    if not benchmark_results:
        raise RuntimeError("No benchmark profile completed successfully.")
    write_benchmark_csv(data_dir / "benchmark_results.csv", benchmark_results)
    write_predictions_csv(data_dir / "extraction_predictions.csv", all_extractions)
    write_json(data_dir / "extractions.json", [asdict(result) for result in all_extractions[:20]])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--gold-size", type=int, default=10)
    parser.add_argument("--gold-path", default="")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--profiles", nargs="+", default=["mistral-fp16", "mistral-int8", "openchat-fp16", "openchat-int8"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--news-classifier-model", default=os.getenv("NEWS_CLASSIFIER_MODEL", DEFAULT_MODEL))
    parser.add_argument("--news-classifier-batch-size", type=int, default=4)
    parser.add_argument("--news-classifier-max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    parser.add_argument("--news-classifier-sanity-check", action="store_true")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--allow-incomplete-gold", action="store_true")
    parser.add_argument("--prepare-annotations", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--rebuild-classifier-cache", action="store_true")

    return parser


def parse_profiles(profiles: list[str]) -> list[str]:
    supported = {"rules", "spacy", "mistral-fp16", "mistral-int8", "openchat-fp16", "openchat-int8"}
    unsupported = set(profiles) - supported
    if unsupported:
        raise ValueError(f"Unsupported profiles: {', '.join(sorted(unsupported))}")

    return profiles


def create_extractor(profile: str, max_new_tokens: int):
    if profile == "rules":
        return RuleBasedNewsExtractor()
    if profile == "spacy":
        return SpacyNewsExtractor()
    alias, precision_mode = profile.rsplit("-", maxsplit=1)

    return TransformersJsonExtractor(MODEL_ALIASES[alias], precision_mode, max_new_tokens=max_new_tokens)


def validate_dataset(dialogs, expected_size: int, allow_synthetic: bool) -> None:
    if len(dialogs) != expected_size or len({dialog.text for dialog in dialogs}) != expected_size:
        raise ValueError("The benchmark dataset must contain the requested number of unique dialogs.")
    if not allow_synthetic and any(dialog.source == "synthetic" for dialog in dialogs):
        raise ValueError("Synthetic dialogs are forbidden in the final benchmark.")


def write_dataset_stats(path: Path, dialogs, filtering_stats: dict | None = None, classifier_info: dict | None = None) -> None:
    sources = Counter(dialog.source for dialog in dialogs)
    payload = {
        "examples": len(dialogs),
        "unique_ids": len({dialog.dialog_id for dialog in dialogs}),
        "unique_texts": len({dialog.text for dialog in dialogs}),
        "sources": dict(sorted(sources.items())),
    }
    if filtering_stats is not None:
        payload["filtering"] = filtering_stats
        payload["classifier"] = classifier_info or {}
    write_json(path, payload)


def write_gold_stats(path: Path, gold_path: Path) -> None:
    gold_labels = load_gold_labels(gold_path)
    label_counts = Counter(label for items in gold_labels.values() for label, _ in items)
    payload = {
        "annotated_dialogs": len(gold_labels),
        "annotations": sum(label_counts.values()),
        "labels": dict(sorted(label_counts.items())),
    }
    write_json(path, payload)


def reset_output_files(data_dir: Path) -> None:
    for filename in ("per_class_metrics.csv", "extraction_errors.csv"):
        path = data_dir / filename
        if path.exists():
            path.unlink()


def clear_cache(cache_dir: Path) -> None:
    if not cache_dir.exists():
        return
    for path in cache_dir.iterdir():
        if path.name != "news_classifier.jsonl" and path.suffix in {".jsonl", ".json"}:
            path.unlink()


def clear_classifier_cache(classifier_cache_path: Path) -> None:
    if classifier_cache_path.exists():
        classifier_cache_path.unlink()


def clear_dataset_cache(dataset_path: Path) -> None:
    if dataset_path.exists():
        dataset_path.unlink()


def clear_dataset_outputs(data_dir: Path) -> None:
    filenames = (
        "benchmark_results.csv",
        "extractions.json",
        "extraction_errors.csv",
        "extraction_predictions.csv",
        "gold_stats.json",
        "per_class_metrics.csv",
    )
    for filename in filenames:
        path = data_dir / filename
        if path.exists():
            path.unlink()


def clear_gpu_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_result(result) -> None:
    print(
        f"{result.extractor}: docs={result.examples}, batch={result.batch_size}, "
        f"docs/sec={result.docs_per_second:.2f}, latency={result.mean_latency_ms:.2f} ms, "
        f"RAM={result.peak_ram_mb:.1f} MB, VRAM={result.peak_vram_mb:.1f} MB, "
        f"micro_f1={result.micro_f1:.3f}, macro_f1={result.macro_f1:.3f}"
    )


def write_json(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)


def write_predictions_csv(path: Path, results: list[ExtractionResult]) -> None:
    fieldnames = ["dialog_id", "extractor", "kind", "label", "value", "start", "end", "confidence", "parse_valid", "error"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for item in result.entities:
                writer.writerow(build_prediction_row(result, "entity", item))
            for item in result.events:
                writer.writerow(build_prediction_row(result, "event", item))
            if not result.parse_valid:
                writer.writerow({"dialog_id": result.dialog_id, "extractor": result.extractor, "parse_valid": False, "error": result.error})


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
        "parse_valid": result.parse_valid,
        "error": result.error,
    }


if __name__ == "__main__":
    main()

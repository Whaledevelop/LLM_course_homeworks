import argparse
import csv
import gc
import json
import shutil
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from benchmark import ExtractionBenchmark, write_benchmark_csv
from dataset import load_news_dialogs
from evaluation import annotation_template_fingerprint, load_gold_labels, write_annotation_template, write_evaluation_csvs
from extractors import RuleBasedNewsExtractor, SpacyNewsExtractor, TransformersJsonExtractor
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
    dataset_path = data_dir / "news_dialogs.jsonl"
    template_path = data_dir / "gold_annotation_template.csv"
    progress_path = data_dir / "annotation_progress.json"
    gold_path = Path(args.gold_path) if args.gold_path else data_dir / "gold_annotations.csv"
    if args.rebuild_cache:
        clear_cache(cache_dir)
    if args.rebuild_dataset:
        clear_dataset_cache(dataset_path)
        clear_dataset_outputs(data_dir)

    dialogs = load_news_dialogs(args.sample_size, args.seed, dataset_path, args.allow_synthetic)
    validate_dataset(dialogs, args.sample_size, args.allow_synthetic)
    write_dataset_stats(data_dir / "dataset_stats.json", dialogs)
    if args.prepare_annotations:
        prepare_annotation_workspace(data_dir, [dialog.dialog_id for dialog in dialogs[:args.gold_size]])
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
    parser.add_argument("--gold-size", type=int, default=20)
    parser.add_argument("--gold-path", default="")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--profiles", nargs="+", default=["mistral-fp16", "mistral-int8", "openchat-fp16", "openchat-int8"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--allow-incomplete-gold", action="store_true")
    parser.add_argument("--prepare-annotations", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--rebuild-dataset", action="store_true")

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


def validate_gold(gold_path: Path, template_path: Path, progress_path: Path, gold_size: int, allow_incomplete: bool) -> None:
    template_dialog_ids = read_template_dialog_ids(template_path)
    if len(template_dialog_ids) != gold_size:
        raise ValueError(f"Annotation template contains {len(template_dialog_ids)} dialogs; exactly {gold_size} are required.")
    target_dialog_ids = template_dialog_ids
    target_dialog_id_set = set(target_dialog_ids)
    gold_labels = load_gold_labels(gold_path)
    unexpected_dialog_ids = set(gold_labels) - target_dialog_id_set
    if unexpected_dialog_ids:
        raise ValueError(f"Gold data contains {len(unexpected_dialog_ids)} dialogs outside the current annotation template.")
    if allow_incomplete:
        return
    progress = read_annotation_progress(progress_path)
    expected_fingerprint = annotation_template_fingerprint(target_dialog_ids)
    if progress.get("template_fingerprint") != expected_fingerprint:
        raise ValueError("Annotation progress does not match the current template.")
    reviewed_dialog_ids = set(progress.get("reviewed_dialog_ids", [])) & target_dialog_id_set
    if len(reviewed_dialog_ids) < gold_size:
        raise ValueError(f"Only {len(reviewed_dialog_ids)} of {gold_size} annotation dialogs are reviewed.")


def prepare_annotation_workspace(data_dir: Path, dialog_ids: list[str]) -> None:
    template_path = data_dir / "gold_annotation_template.csv"
    gold_path = data_dir / "gold_annotations.csv"
    progress_path = data_dir / "annotation_progress.json"
    existing_dialog_ids = read_template_dialog_ids(template_path) if template_path.exists() else []
    if existing_dialog_ids != dialog_ids:
        backup_annotation_files(data_dir, (template_path, gold_path, progress_path))
        write_annotation_template(template_path, dialog_ids)
        write_empty_gold(gold_path)
        write_annotation_progress(progress_path, dialog_ids, [])
        return
    progress = read_annotation_progress(progress_path)
    reviewed_dialog_ids = [dialog_id for dialog_id in progress.get("reviewed_dialog_ids", []) if dialog_id in set(dialog_ids)]
    write_annotation_progress(progress_path, dialog_ids, reviewed_dialog_ids)


def read_template_dialog_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        return [row["dialog_id"].strip() for row in reader if row.get("dialog_id", "").strip()]


def read_annotation_progress(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:

        return json.load(file)


def write_annotation_progress(path: Path, dialog_ids: list[str], reviewed_dialog_ids: list[str]) -> None:
    reviewed = [dialog_id for dialog_id in dialog_ids if dialog_id in set(reviewed_dialog_ids)]
    write_json(
        path,
        {"template_fingerprint": annotation_template_fingerprint(dialog_ids), "reviewed_dialog_ids": reviewed},
    )


def write_empty_gold(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["dialog_id", "label", "value"])
        writer.writeheader()


def backup_annotation_files(data_dir: Path, paths: tuple[Path, ...]) -> None:
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        return
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup_dir = data_dir / "annotation_backups" / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in existing_paths:
        shutil.copy2(path, backup_dir / path.name)


def write_dataset_stats(path: Path, dialogs) -> None:
    sources = Counter(dialog.source for dialog in dialogs)
    payload = {
        "examples": len(dialogs),
        "unique_ids": len({dialog.dialog_id for dialog in dialogs}),
        "unique_texts": len({dialog.text for dialog in dialogs}),
        "sources": dict(sorted(sources.items())),
    }
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
        if path.suffix in {".jsonl", ".json"}:
            path.unlink()


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

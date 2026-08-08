import csv
import hashlib
import json
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

from evaluation import EvaluationReport, evaluate
from schemas import BenchmarkResult, ExtractedItem, ExtractionResult, NewsDialog


class ExtractionBenchmark:
    def __init__(self, cache_dir: Path, gold_path: Path, evaluation_dialog_ids: set[str] | None = None) -> None:
        self._cache_dir = cache_dir
        self._gold_path = gold_path
        self._evaluation_dialog_ids = evaluation_dialog_ids
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def run(self, extractor, dialogs: list[NewsDialog], batch_size: int) -> tuple[BenchmarkResult, list[ExtractionResult], EvaluationReport]:
        fingerprint = build_cache_fingerprint(extractor, dialogs, batch_size)
        result_path = self._cache_dir / f"{fingerprint}.jsonl"
        metrics_path = self._cache_dir / f"{fingerprint}.metrics.json"
        if result_path.exists() and metrics_path.exists():
            results = read_results(result_path)
            with metrics_path.open("r", encoding="utf-8") as file:
                benchmark_result = BenchmarkResult(**json.load(file))
            report = evaluate(results, self._gold_path, self._evaluation_dialog_ids)
            benchmark_result = replace(
                benchmark_result,
                precision=report.precision,
                recall=report.recall,
                f1=report.micro_f1,
                micro_f1=report.micro_f1,
                macro_f1=report.macro_f1,
            )

            return benchmark_result, results, report

        process = psutil.Process() if psutil else None
        reset_peak_vram()
        latencies = []
        results = []
        started_at = time.perf_counter()
        peak_ram_mb = current_ram_mb(process)
        for batch in chunked(dialogs, batch_size):
            batch_started_at = time.perf_counter()
            batch_results = extractor.extract_batch(batch)
            elapsed = time.perf_counter() - batch_started_at
            latencies.extend([elapsed / len(batch)] * len(batch))
            results.extend(batch_results)
            peak_ram_mb = max(peak_ram_mb, current_ram_mb(process))
        total_seconds = time.perf_counter() - started_at
        total_chars = sum(len(dialog.text) for dialog in dialogs)
        report = evaluate(results, self._gold_path, self._evaluation_dialog_ids)
        valid_json_rate = sum(result.parse_valid for result in results) / len(results) if results else 0.0
        benchmark_result = BenchmarkResult(
            extractor=extractor.name,
            model=extractor.model_name,
            precision_mode=extractor.precision_mode,
            dataset_source=dataset_source(dialogs),
            examples=len(dialogs),
            unique_examples=len({dialog.text for dialog in dialogs}),
            batch_size=batch_size,
            load_seconds=extractor.load_seconds,
            total_seconds=total_seconds,
            docs_per_second=len(dialogs) / total_seconds if total_seconds else 0.0,
            chars_per_second=total_chars / total_seconds if total_seconds else 0.0,
            estimated_tokens_per_second=(total_chars / 4) / total_seconds if total_seconds else 0.0,
            mean_latency_ms=statistics.mean(latencies) * 1000 if latencies else 0.0,
            p95_latency_ms=percentile(latencies, 0.95) * 1000 if latencies else 0.0,
            peak_ram_mb=peak_ram_mb,
            peak_vram_mb=peak_vram_mb(),
            valid_json_rate=valid_json_rate,
            precision=report.precision,
            recall=report.recall,
            f1=report.micro_f1,
            micro_f1=report.micro_f1,
            macro_f1=report.macro_f1,
        )
        write_results(result_path, results)
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(asdict(benchmark_result), file, ensure_ascii=False, indent=2)

        return benchmark_result, results, report


def build_cache_fingerprint(extractor, dialogs: list[NewsDialog], batch_size: int) -> str:
    payload = {
        "extractor": extractor.name,
        "model": extractor.model_name,
        "precision": extractor.precision_mode,
        "revision": getattr(extractor, "revision", ""),
        "prompt_version": extractor.prompt_version,
        "generation_config": extractor.generation_config,
        "batch_size": batch_size,
        "dialogs": [(dialog.dialog_id, hashlib.sha256(dialog.text.encode("utf-8")).hexdigest()) for dialog in dialogs],
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)

    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]


def chunked(items: list[NewsDialog], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def percentile(values: list[float], rank: float) -> float:
    if not values:
        return 0.0
    ordered_values = sorted(values)
    index = min(len(ordered_values) - 1, round((len(ordered_values) - 1) * rank))

    return ordered_values[index]


def current_ram_mb(process=None) -> float:
    if not psutil:
        return 0.0
    active_process = process or psutil.Process()

    return active_process.memory_info().rss / 1024 / 1024


def reset_peak_vram() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_mb() -> float:
    try:
        import torch
    except ImportError:
        return 0.0
    if not torch.cuda.is_available():
        return 0.0

    return torch.cuda.max_memory_allocated() / 1024 / 1024


def dataset_source(dialogs: list[NewsDialog]) -> str:
    sources = sorted({dialog.source for dialog in dialogs})

    return "+".join(sources)


def write_results(path: Path, results: list[ExtractionResult]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def read_results(path: Path) -> list[ExtractionResult]:
    results = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            row["entities"] = [ExtractedItem(**item) for item in row.get("entities", [])]
            row["events"] = [ExtractedItem(**item) for item in row.get("events", [])]
            results.append(ExtractionResult(**row))

    return results


def write_benchmark_csv(path: Path, results: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

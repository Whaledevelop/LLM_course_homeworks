import csv
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

from evaluation import evaluate
from schemas import BenchmarkResult, ExtractionResult, NewsDialog


class ExtractionBenchmark:
    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def run(self, extractor, dialogs: list[NewsDialog], batch_size: int) -> tuple[BenchmarkResult, list[ExtractionResult]]:
        cache_path = self._cache_dir / f"{extractor.name}_{len(dialogs)}_{batch_size}.jsonl"
        cached_results = read_results(cache_path)
        if cached_results:
            precision, recall, f1 = evaluate(cached_results)
            benchmark_result = BenchmarkResult(extractor=extractor.name, examples=len(cached_results), batch_size=batch_size, total_seconds=0.0, docs_per_second=0.0, chars_per_second=0.0, estimated_tokens_per_second=0.0, mean_latency_ms=0.0, p95_latency_ms=0.0, peak_rss_mb=current_rss_mb(), precision=precision, recall=recall, f1=f1)

            return benchmark_result, cached_results

        process = psutil.Process() if psutil else None
        latencies = []
        results = []
        started_at = time.perf_counter()
        peak_rss_mb = current_rss_mb(process)
        for batch in chunked(dialogs, batch_size):
            batch_started_at = time.perf_counter()
            batch_results = extractor.extract_batch(batch)
            elapsed = time.perf_counter() - batch_started_at
            latencies.extend([elapsed / len(batch)] * len(batch))
            results.extend(batch_results)
            peak_rss_mb = max(peak_rss_mb, current_rss_mb(process))
        total_seconds = time.perf_counter() - started_at
        total_chars = sum(len(dialog.text) for dialog in dialogs)
        estimated_tokens = total_chars / 4
        precision, recall, f1 = evaluate(results)
        benchmark_result = BenchmarkResult(extractor=extractor.name, examples=len(dialogs), batch_size=batch_size, total_seconds=total_seconds, docs_per_second=len(dialogs) / total_seconds if total_seconds else 0.0, chars_per_second=total_chars / total_seconds if total_seconds else 0.0, estimated_tokens_per_second=estimated_tokens / total_seconds if total_seconds else 0.0, mean_latency_ms=statistics.mean(latencies) * 1000 if latencies else 0.0, p95_latency_ms=percentile(latencies, 0.95) * 1000 if latencies else 0.0, peak_rss_mb=peak_rss_mb, precision=precision, recall=recall, f1=f1)
        write_results(cache_path, results)

        return benchmark_result, results


def chunked(items: list[NewsDialog], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def percentile(values: list[float], rank: float) -> float:
    if not values:
        return 0.0
    ordered_values = sorted(values)
    index = min(len(ordered_values) - 1, round((len(ordered_values) - 1) * rank))

    return ordered_values[index]


def current_rss_mb(process=None) -> float:
    if not psutil:
        return 0.0
    active_process = process or psutil.Process()

    return active_process.memory_info().rss / 1024 / 1024


def write_results(path: Path, results: list[ExtractionResult]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def read_results(path: Path) -> list[ExtractionResult]:
    if not path.exists():
        return []

    from schemas import ExtractedItem

    results = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            entities = [ExtractedItem(**item) for item in row.get("entities", [])]
            events = [ExtractedItem(**item) for item in row.get("events", [])]
            results.append(ExtractionResult(dialog_id=row["dialog_id"], extractor=row["extractor"], entities=entities, events=events, relations=row.get("relations", []), raw_response=row.get("raw_response", "")))

    return results


def write_benchmark_csv(path: Path, results: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

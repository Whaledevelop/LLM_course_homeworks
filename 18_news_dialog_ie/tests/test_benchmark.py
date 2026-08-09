from benchmark import ExtractionBenchmark, build_cache_fingerprint
from extractors import RuleBasedNewsExtractor
from schemas import NewsDialog


def test_cache_fingerprint_changes_with_batch_and_text() -> None:
    extractor = RuleBasedNewsExtractor()
    dialogs = [NewsDialog("1", "test", "Reuters reported news")]
    first = build_cache_fingerprint(extractor, dialogs, 1)
    second = build_cache_fingerprint(extractor, dialogs, 2)
    third = build_cache_fingerprint(extractor, [NewsDialog("1", "test", "BBC reported news")], 1)

    assert len({first, second, third}) == 3


def test_cached_benchmark_preserves_metrics(tmp_path) -> None:
    gold_path = tmp_path / "gold.csv"
    gold_path.write_text("dialog_id,label,value\n1,ORG,Reuters\n", encoding="utf-8")
    dialogs = [NewsDialog("1", "test", "Reuters reported news")]
    extractor = RuleBasedNewsExtractor()
    benchmark = ExtractionBenchmark(tmp_path / "cache", gold_path)
    first, _, _ = benchmark.run(extractor, dialogs, 1)
    second, _, _ = benchmark.run(extractor, dialogs, 1)

    assert first.total_seconds > 0
    assert second.total_seconds == first.total_seconds
    assert second.docs_per_second == first.docs_per_second


def test_benchmark_reports_every_ten_dialogs_and_completion(tmp_path) -> None:
    gold_path = tmp_path / "gold.csv"
    gold_path.write_text("dialog_id,label,value\n", encoding="utf-8")
    dialogs = [NewsDialog(str(index), "test", "No entities") for index in range(23)]
    progress = []

    ExtractionBenchmark(tmp_path / "cache", gold_path).run(
        RuleBasedNewsExtractor(),
        dialogs,
        8,
        lambda processed, total: progress.append((processed, total)),
    )

    assert progress == [(10, 23), (20, 23), (23, 23)]

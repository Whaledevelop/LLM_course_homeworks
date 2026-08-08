import json

import pytest

import news_classifier
from news_classifier import NewsClassifier, parse_classification, select_device


def test_parse_classification_is_strict() -> None:
    assert parse_classification(" NEWS\n") == "NEWS"
    assert parse_classification("NOT_NEWS") == "NOT_NEWS"
    assert parse_classification("The answer is NEWS") == "INVALID"
    assert parse_classification("") == "INVALID"


def test_cache_hit_does_not_run_generator(tmp_path) -> None:
    generator_calls = []

    def loader(model_id: str):
        def generate(prompts: list[str]) -> list[str]:
            generator_calls.append(prompts)
            return ["NEWS"] * len(prompts)

        return generate, "cpu", 0.25

    cache_path = tmp_path / "classifier.jsonl"
    classifier = NewsClassifier(cache_path, "model", 8, loader)
    first = classifier.classify("one", "Reuters reported an election.")
    second = classifier.classify("two", "Reuters reported an election.")

    assert first == ("NEWS", False)
    assert second == ("NEWS", True)
    assert len(generator_calls) == 1


def test_changed_text_model_and_prompt_do_not_reuse_cache(tmp_path, monkeypatch) -> None:
    generator_calls = []

    def loader(model_id: str):
        def generate(prompts: list[str]) -> list[str]:
            generator_calls.extend(prompts)
            return ["NOT_NEWS"] * len(prompts)

        return generate, "cpu", 0.1

    cache_path = tmp_path / "classifier.jsonl"
    first = NewsClassifier(cache_path, "model-a", 8, loader)
    first.classify("one", "First text")
    first.classify("one", "Changed text")
    second = NewsClassifier(cache_path, "model-b", 8, loader)
    second.classify("one", "First text")
    monkeypatch.setattr(news_classifier, "PROMPT_VERSION", "local-news-classifier-v2")
    third = NewsClassifier(cache_path, "model-a", 8, loader)
    third.classify("one", "First text")

    assert len(generator_calls) == 4


def test_invalid_output_is_cached(tmp_path) -> None:
    cache_path = tmp_path / "classifier.jsonl"
    loader = lambda model_id: (lambda prompts: ["maybe"] * len(prompts), "cpu", 0.1)
    classifier = NewsClassifier(cache_path, "model", 8, loader)

    assert classifier.classify("one", "text") == ("INVALID", False)
    record = json.loads(cache_path.read_text(encoding="utf-8"))
    assert record["classification"] == "INVALID"


def test_classifier_loads_model_once_for_multiple_batches(tmp_path) -> None:
    loader_calls = []
    generation_calls = []

    def loader(model_id: str):
        loader_calls.append(model_id)

        def generate(prompts: list[str]) -> list[str]:
            generation_calls.append(len(prompts))
            return ["NEWS"] * len(prompts)

        return generate, "cpu", 0.4

    classifier = NewsClassifier(tmp_path / "classifier.jsonl", "model", 2, loader)
    classifier.classify_batch([("one", "first"), ("two", "second")])
    classifier.classify("three", "third")

    assert loader_calls == ["model"]
    assert generation_calls == [2, 1]
    assert classifier.device == "cpu"
    assert classifier.load_seconds == 0.4


def test_batch_output_preserves_order_and_cache_flags(tmp_path) -> None:
    responses = iter([["NEWS"], ["NOT_NEWS", "invalid"]])
    loader = lambda model_id: (lambda prompts: next(responses), "cpu", 0.1)
    classifier = NewsClassifier(tmp_path / "classifier.jsonl", "model", 8, loader)
    classifier.classify("cached", "cached text")

    results = classifier.classify_batch(
        [("one", "first"), ("cached-again", "cached text"), ("two", "second")]
    )

    assert results == [("NOT_NEWS", False), ("NEWS", True), ("INVALID", False)]


def test_device_selection() -> None:
    assert select_device(False) == "cpu"
    assert select_device(True) == "cuda"


def test_batch_size_must_be_positive(tmp_path) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        NewsClassifier(tmp_path / "classifier.jsonl", "model", 0)

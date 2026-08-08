import json

import pytest
import requests

import news_classifier
from news_classifier import NewsClassifier, parse_classification, validate_classifier_settings


class FakeResponse:
    def __init__(self, content: str = "NEWS", payload: dict | None = None) -> None:
        self._content = content
        self._payload = payload

    def raise_for_status(self) -> None:
        return

    def json(self) -> dict:
        return self._payload or {"choices": [{"message": {"content": self._content}}]}


def test_parse_classification_is_strict() -> None:
    assert parse_classification(" NEWS\n") == "NEWS"
    assert parse_classification("NOT_NEWS") == "NOT_NEWS"
    assert parse_classification("The answer is NEWS") == "INVALID"
    assert parse_classification("") == "INVALID"


def test_missing_settings_raise_clear_error(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NEWS_CLASSIFIER_BASE_URL", raising=False)
    monkeypatch.delenv("NEWS_CLASSIFIER_MODEL", raising=False)

    with pytest.raises(RuntimeError, match="NEWS_CLASSIFIER_BASE_URL and NEWS_CLASSIFIER_MODEL"):
        validate_classifier_settings()


def test_cache_hit_does_not_call_api(tmp_path) -> None:
    calls = []

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeResponse("NEWS")

    cache_path = tmp_path / "classifier.jsonl"
    classifier = NewsClassifier(cache_path, "http://localhost/v1", "", "model", post)
    first = classifier.classify("one", "Reuters reported an election.")
    second = classifier.classify("two", "Reuters reported an election.")

    assert first == ("NEWS", False)
    assert second == ("NEWS", True)
    assert len(calls) == 1


def test_changed_text_model_and_prompt_do_not_reuse_cache(tmp_path, monkeypatch) -> None:
    calls = []

    def post(*args, **kwargs):
        calls.append(1)
        return FakeResponse("NOT_NEWS")

    cache_path = tmp_path / "classifier.jsonl"
    first = NewsClassifier(cache_path, "http://localhost/v1", "", "model-a", post)
    first.classify("one", "First text")
    first.classify("one", "Changed text")
    second = NewsClassifier(cache_path, "http://localhost/v1", "", "model-b", post)
    second.classify("one", "First text")
    monkeypatch.setattr(news_classifier, "PROMPT_VERSION", "news-classifier-v2")
    third = NewsClassifier(cache_path, "http://localhost/v1", "", "model-a", post)
    third.classify("one", "First text")

    assert len(calls) == 4


def test_invalid_output_is_cached(tmp_path) -> None:
    cache_path = tmp_path / "classifier.jsonl"
    classifier = NewsClassifier(cache_path, "http://localhost/v1", "", "model", lambda *args, **kwargs: FakeResponse("maybe"))

    assert classifier.classify("one", "text") == ("INVALID", False)
    record = json.loads(cache_path.read_text(encoding="utf-8"))
    assert record["classification"] == "INVALID"


def test_api_failure_does_not_modify_cache(tmp_path) -> None:
    def post(*args, **kwargs):
        raise requests.ConnectionError("offline")

    cache_path = tmp_path / "classifier.jsonl"
    classifier = NewsClassifier(cache_path, "http://localhost/v1", "", "model", post)

    with pytest.raises(RuntimeError, match="API request failed"):
        classifier.classify("one", "text")
    assert not cache_path.exists()


def test_invalid_response_schema_does_not_modify_cache(tmp_path) -> None:
    cache_path = tmp_path / "classifier.jsonl"
    classifier = NewsClassifier(
        cache_path,
        "http://localhost/v1",
        "",
        "model",
        lambda *args, **kwargs: FakeResponse(payload={"unexpected": []}),
    )

    with pytest.raises(RuntimeError, match="invalid response schema"):
        classifier.classify("one", "text")
    assert not cache_path.exists()


def test_non_string_response_does_not_modify_cache(tmp_path) -> None:
    cache_path = tmp_path / "classifier.jsonl"
    payload = {"choices": [{"message": {"content": ["NEWS"]}}]}
    classifier = NewsClassifier(
        cache_path,
        "http://localhost/v1",
        "",
        "model",
        lambda *args, **kwargs: FakeResponse(payload=payload),
    )

    with pytest.raises(RuntimeError, match="invalid response schema"):
        classifier.classify("one", "text")
    assert not cache_path.exists()

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests
from dotenv import load_dotenv


PROMPT_VERSION = "news-classifier-v1"
SYSTEM_PROMPT = (
    "Classify the dialog. Reply only NEWS or NOT_NEWS. NEWS mainly discusses or asks to analyze a real or "
    "supposed current news event, news article, or public, political, or economic development. NOT_NEWS includes "
    "fiction, programming, roleplay, creative writing, marketing, generic education, and incidental news words."
)
VALID_CLASSIFICATIONS = {"NEWS", "NOT_NEWS"}


class NewsClassifier:
    def __init__(
        self,
        cache_path: Path,
        base_url: str = "",
        api_key: str | None = None,
        model: str = "",
        post: Callable | None = None,
    ) -> None:
        load_dotenv()
        self.base_url = (base_url or os.getenv("NEWS_CLASSIFIER_BASE_URL", "")).rstrip("/")
        self.api_key = os.getenv("NEWS_CLASSIFIER_API_KEY", "") if api_key is None else api_key
        self.model = model or os.getenv("NEWS_CLASSIFIER_MODEL", "")
        validate_classifier_settings(self.base_url, self.model)
        self._cache_path = cache_path
        self._post = post or requests.post
        self._cache = self._load_cache()

    def classify(self, dialog_id: str, text: str) -> tuple[str, bool]:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cache_key = self._cache_key(text_hash)
        cached = self._cache.get(cache_key)
        if cached:
            return cached["classification"], True
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "temperature": 0,
            "max_tokens": 4,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            response = self._post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise RuntimeError(f"News classifier API request failed: {error}") from error
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise RuntimeError("News classifier API returned an invalid response schema.") from error
        if not isinstance(content, str):
            raise RuntimeError("News classifier API returned an invalid response schema.")
        classification = parse_classification(content)
        record = {
            "dialog_id": dialog_id,
            "text_hash": text_hash,
            "model": self.model,
            "prompt_version": PROMPT_VERSION,
            "classification": classification,
            "classified_at": datetime.now(timezone.utc).isoformat(),
        }
        self._append_cache(record)
        self._cache[cache_key] = record

        return classification, False

    def _cache_key(self, text_hash: str) -> str:
        return f"{text_hash}:{self.model}:{PROMPT_VERSION}"

    def _load_cache(self) -> dict[str, dict]:
        if not self._cache_path.exists():
            return {}
        cache = {}
        with self._cache_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    record = json.loads(line)
                    cache_key = (
                        f"{record['text_hash']}:{record['model']}:"
                        f"{record.get('prompt_version', PROMPT_VERSION)}"
                    )
                    cache[cache_key] = record
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    raise RuntimeError(f"Invalid news classifier cache at line {line_number}.") from error

        return cache

    def _append_cache(self, record: dict) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_classification(value: str) -> str:
    normalized = value.strip()

    return normalized if normalized in VALID_CLASSIFICATIONS else "INVALID"


def validate_classifier_settings(base_url: str = "", model: str = "") -> None:
    load_dotenv()
    active_base_url = base_url or os.getenv("NEWS_CLASSIFIER_BASE_URL", "")
    active_model = model or os.getenv("NEWS_CLASSIFIER_MODEL", "")
    if not active_base_url or not active_model:
        raise RuntimeError(
            "News classifier settings are missing. Set NEWS_CLASSIFIER_BASE_URL and NEWS_CLASSIFIER_MODEL. "
            "NEWS_CLASSIFIER_API_KEY may be empty for a local endpoint."
        )

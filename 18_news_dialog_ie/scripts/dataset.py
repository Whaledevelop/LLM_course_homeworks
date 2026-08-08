import hashlib
import json
import random
import re
from pathlib import Path

from news_classifier import NewsClassifier
from schemas import NewsDialog


NEWS_SOURCE_PATTERN = re.compile(
    r"\b(?:Reuters|Associated Press|AP News|BBC|CNN|The Guardian)\b|\b(?:according to|reported by)\b",
    re.IGNORECASE,
)
NEWS_INTENT_PATTERN = re.compile(
    r"\b(?:summari[sz]e (?:this |the )?(?:news|article|report)|what happened|latest developments?|"
    r"explain (?:this |the )?headline|news report|breaking news)\b",
    re.IGNORECASE,
)
EVENT_PATTERN = re.compile(
    r"\b(?:elections?|summits?|earthquakes?|floods?|attacks?|investigations?|sanctions?|layoffs?|"
    r"protests?|court rulings?|mergers?|acquisitions?|strikes?|conflicts?|ceasefires?|arrests?|"
    r"resignations?|disasters?|explosions?|wildfires?|pandemics?|referendums?)\b",
    re.IGNORECASE,
)
DATE_PATTERN = re.compile(
    r"\b(?:19|20)\d{2}\b|\b\d{4}-\d{2}-\d{2}\b|"
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b",
    re.IGNORECASE,
)
REPORTING_PATTERN = re.compile(
    r"\b(?:announced|reported|confirmed|warned|approved|declared|said|stated|revealed|issued)\b",
    re.IGNORECASE,
)
CAPITALIZED_PHRASE_PATTERN = re.compile(r"\b(?:[A-Z][a-z]{2,})(?:\s+[A-Z][a-z]{2,})+\b")
PROGRAMMING_PATTERNS = (
    re.compile(r"```(?:python|csharp|c#|java|javascript|typescript|cpp|sql)?", re.IGNORECASE),
    re.compile(r"\b(?:import|def|class|public static|namespace|using System|StackTrace|Traceback|Exception)\b"),
    re.compile(r"\b(?:compiler error|compile error|runtime error|KeyError|TypeError|NullReferenceException)\b", re.IGNORECASE),
)
EXCLUDED_PATTERNS = (
    re.compile(r"\b(?:ignore previous instructions|jailbreak|DAN mode|unfiltered model|free of all restrictions)\b", re.IGNORECASE),
    re.compile(r"\b(?:roleplay as|you are now|act as an? (?:unrestricted|unfiltered|fictional))\b", re.IGNORECASE),
    re.compile(r"\b(?:porn websites?|underage content|explicit sexual content|EroticaChan)\b", re.IGNORECASE),
    re.compile(r"\b(?:write (?:an? )?ad copy|marketing campaign|product promotion|voice-overs? (?:related to|for) (?:the )?theme)\b", re.IGNORECASE),
)


def load_news_dialogs(
    sample_size: int,
    seed: int,
    output_path: Path,
    classifier_cache_path: Path,
    classifier_model: str,
    classifier_batch_size: int,
    allow_synthetic: bool = False,
    progress_callback=None,
) -> tuple[list[NewsDialog], dict | None, dict | None]:
    cached_dialogs = read_jsonl(output_path)
    cached_dialogs = unique_dialogs(cached_dialogs)
    if len(cached_dialogs) >= sample_size and (allow_synthetic or all(dialog.source != "synthetic" for dialog in cached_dialogs[:sample_size])):
        return cached_dialogs[:sample_size], None, None

    classifier = NewsClassifier(classifier_cache_path, classifier_model, classifier_batch_size)
    emit_progress(
        progress_callback,
        "start",
        {
            "target": sample_size,
            "model": classifier.model,
            "device": classifier.device,
            "batch_size": classifier.batch_size,
        },
    )
    dialogs, filtering_stats = stream_wildchat_news(sample_size, seed, classifier, progress_callback)
    dialogs = unique_dialogs(dialogs)
    if len(dialogs) < sample_size:
        raise RuntimeError(f"Expected {sample_size} unique WildChat dialogs, received {len(dialogs)}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, dialogs[:sample_size])

    classifier_info = {
        "model": classifier.model,
        "device": classifier.device,
        "classifier_load_seconds": classifier.load_seconds,
    }

    return dialogs[:sample_size], filtering_stats, classifier_info


def stream_wildchat_news(sample_size: int, seed: int, classifier: NewsClassifier, progress_callback=None) -> tuple[list[NewsDialog], dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("The datasets package is required to load WildChat.")

    dataset = load_dataset("allenai/WildChat-1M", split="train", streaming=True)

    return collect_news_dialogs(dataset, sample_size, seed, classifier, progress_callback)


def collect_news_dialogs(rows, sample_size: int, seed: int, classifier, progress_callback=None) -> tuple[list[NewsDialog], dict]:
    selected = []
    candidates = []
    seen_ids = set()
    seen_text_hashes = set()
    stats = {
        "rows_seen": 0,
        "stage1_passed": 0,
        "llm_classified": 0,
        "llm_news": 0,
        "llm_not_news": 0,
        "llm_invalid": 0,
        "classifier_cache_hits": 0,
    }
    for row_index, row in enumerate(rows):
        stats["rows_seen"] += 1
        if stats["rows_seen"] % 1000 == 0:
            emit_progress(progress_callback, "scan", {**stats, "target": sample_size})
        text = flatten_conversation(row)
        language = str(row.get("language") or row.get("lang") or "").lower()
        if language and language not in {"en", "english"}:
            continue
        if not is_english_text(text):
            continue
        dialog_id = str(row.get("conversation_hash") or row.get("conversation_id") or row_index)
        text_hash = hashlib.sha256(" ".join(text.lower().split()).encode("utf-8")).hexdigest()
        if dialog_id in seen_ids or text_hash in seen_text_hashes:
            continue
        seen_ids.add(dialog_id)
        seen_text_hashes.add(text_hash)
        if not passes_news_prefilter(text):
            continue
        stats["stage1_passed"] += 1
        created_at = str(row.get("timestamp") or "")
        candidates.append(NewsDialog(dialog_id=dialog_id, source="allenai/WildChat-1M", text=text, created_at=created_at))
        if len(candidates) >= classifier.batch_size:
            classify_candidates(candidates, classifier, selected, stats, sample_size, progress_callback)
            candidates.clear()
            if len(selected) >= sample_size:
                break
    if candidates and len(selected) < sample_size:
        classify_candidates(candidates, classifier, selected, stats, sample_size, progress_callback)

    random.Random(seed).shuffle(selected)

    return selected, stats


def classify_candidates(candidates, classifier, selected: list[NewsDialog], stats: dict, sample_size: int, progress_callback=None) -> None:
    classifications = classifier.classify_batch([(dialog.dialog_id, dialog.text) for dialog in candidates])
    for dialog, (classification, cache_hit) in zip(candidates, classifications):
        stats["llm_classified"] += 1
        stats["classifier_cache_hits"] += int(cache_hit)
        if classification == "NOT_NEWS":
            stats["llm_not_news"] += 1
        elif classification != "NEWS":
            stats["llm_invalid"] += 1
        else:
            stats["llm_news"] += 1
            if len(selected) < sample_size:
                selected.append(dialog)
        emit_progress(
            progress_callback,
            "classification",
            {
                "index": stats["llm_classified"],
                "classification": classification,
                "cache_hit": cache_hit,
                "news_collected": len(selected),
                "target": sample_size,
                "text": dialog.text,
            },
        )


def emit_progress(progress_callback, event: str, payload: dict) -> None:
    if progress_callback is not None:
        progress_callback(event, payload)


def passes_news_prefilter(text: str) -> bool:
    if has_excluded_content(text):
        return False
    score, features = news_dialog_score(text)
    has_news_signal = bool({"source", "intent", "event", "reporting"} & features)

    return score >= 2 and has_news_signal


def is_news_dialog(text: str) -> bool:
    return passes_news_prefilter(text)


def flatten_conversation(row: dict) -> str:
    conversations = row.get("conversation") or row.get("messages") or []
    parts = []
    for message in conversations:
        role = str(message.get("role") or message.get("from") or "message").title()
        content = str(message.get("content") or message.get("value") or "")
        if content:
            parts.append(f"{role}: {content}")

    return "\n".join(parts)


def news_dialog_score(text: str) -> tuple[int, set[str]]:
    score = 0
    features = set()
    feature_patterns = (
        ("source", 2, NEWS_SOURCE_PATTERN),
        ("intent", 2, NEWS_INTENT_PATTERN),
        ("event", 1, EVENT_PATTERN),
        ("date", 1, DATE_PATTERN),
        ("reporting", 1, REPORTING_PATTERN),
    )
    for feature, points, pattern in feature_patterns:
        if pattern.search(text):
            features.add(feature)
            score += points
    capitalized_phrases = {
        match.group(0)
        for match in CAPITALIZED_PHRASE_PATTERN.finditer(text)
        if match.group(0) not in {"User Assistant", "Assistant User"}
    }
    if len(capitalized_phrases) >= 2:
        features.add("entities")
        score += 1

    return score, features


def has_excluded_content(text: str) -> bool:
    if any(pattern.search(text) for pattern in EXCLUDED_PATTERNS):
        return True
    programming_signals = sum(bool(pattern.search(text)) for pattern in PROGRAMMING_PATTERNS)

    return programming_signals >= 2


def is_english_text(text: str) -> bool:
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return False
    ascii_letters = sum(character.isascii() for character in letters)

    return ascii_letters / len(letters) >= 0.9


def unique_dialogs(dialogs: list[NewsDialog]) -> list[NewsDialog]:
    seen_ids = set()
    seen_texts = set()
    result = []
    for dialog in dialogs:
        text_fingerprint = hashlib.sha256(" ".join(dialog.text.lower().split()).encode("utf-8")).hexdigest()
        if dialog.dialog_id in seen_ids or text_fingerprint in seen_texts:
            continue
        seen_ids.add(dialog.dialog_id)
        seen_texts.add(text_fingerprint)
        result.append(dialog)

    return result


def read_jsonl(path: Path) -> list[NewsDialog]:
    if not path.exists():
        return []

    dialogs = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            dialogs.append(NewsDialog(**row))

    return dialogs


def write_jsonl(path: Path, dialogs: list[NewsDialog]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for dialog in dialogs:
            file.write(json.dumps(dialog.__dict__, ensure_ascii=False) + "\n")

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Iterable

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


SAMPLE_DIALOGS = [
    NewsDialog(
        dialog_id="sample-001",
        source="synthetic",
        text=(
            "User: Reuters reported that President Joe Biden met NATO Secretary General Jens Stoltenberg "
            "in Washington on April 4, 2024. What was the impact?\n"
            "Assistant: The meeting focused on Ukraine aid and strengthened NATO coordination before the summit."
        ),
        created_at="2024-04-04",
    ),
    NewsDialog(
        dialog_id="sample-002",
        source="synthetic",
        text=(
            "User: BBC says Apple announced a $110 billion share buyback after quarterly earnings in California "
            "on May 2, 2024. Summarize the event.\n"
            "Assistant: The announcement boosted investor confidence and lifted Apple shares in after-hours trading."
        ),
        created_at="2024-05-02",
    ),
    NewsDialog(
        dialog_id="sample-003",
        source="synthetic",
        text=(
            "User: AP News reported an earthquake near Hualien, Taiwan on April 3, 2024, damaging buildings "
            "and disrupting transport. Who responded?\n"
            "Assistant: Local authorities and rescue teams began evacuations and infrastructure inspections."
        ),
        created_at="2024-04-03",
    ),
    NewsDialog(
        dialog_id="sample-004",
        source="synthetic",
        text=(
            "User: The Guardian reported that Tesla will cut jobs in Texas and California in 2024. "
            "What is the business impact?\n"
            "Assistant: The layoffs signal cost pressure and may affect production planning."
        ),
        created_at="2024-04-15",
    ),
    NewsDialog(
        dialog_id="sample-005",
        source="synthetic",
        text=(
            "User: CNN reported that the European Union opened an investigation into Microsoft in Brussels "
            "on June 25, 2024. What happened?\n"
            "Assistant: Regulators examined whether bundling practices harmed competition in cloud and software markets."
        ),
        created_at="2024-06-25",
    ),
]


def load_news_dialogs(sample_size: int, seed: int, output_path: Path, allow_synthetic: bool = False) -> list[NewsDialog]:
    cached_dialogs = read_jsonl(output_path)
    cached_dialogs = unique_dialogs(cached_dialogs)
    if len(cached_dialogs) >= sample_size and (allow_synthetic or all(dialog.source != "synthetic" for dialog in cached_dialogs[:sample_size])):
        return cached_dialogs[:sample_size]

    try:
        dialogs = list(stream_wildchat_news(sample_size, seed))
    except Exception as error:
        if not allow_synthetic:
            raise RuntimeError("WildChat loading failed. Use --allow-synthetic only for smoke tests.") from error
        dialogs = []
    if len(dialogs) < sample_size and allow_synthetic:
        dialogs.extend(build_fallback_dialogs(sample_size - len(dialogs), seed, len(dialogs)))
    dialogs = unique_dialogs(dialogs)
    if len(dialogs) < sample_size:
        raise RuntimeError(f"Expected {sample_size} unique WildChat dialogs, received {len(dialogs)}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, dialogs[:sample_size])

    return dialogs[:sample_size]


def stream_wildchat_news(sample_size: int, seed: int) -> Iterable[NewsDialog]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("The datasets package is required to load WildChat.")

    random.seed(seed)
    dataset = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    selected = []
    for row_index, row in enumerate(dataset):
        text = flatten_conversation(row)
        language = str(row.get("language") or row.get("lang") or "").lower()
        if language and language not in {"en", "english"}:
            continue
        if not is_english_text(text) or not is_news_dialog(text):
            continue
        dialog_id = str(row.get("conversation_hash") or row.get("conversation_id") or row_index)
        created_at = str(row.get("timestamp") or "")
        selected.append(NewsDialog(dialog_id=dialog_id, source="allenai/WildChat-1M", text=text, created_at=created_at))
        if len(selected) >= sample_size:
            break

    random.shuffle(selected)

    return selected


def flatten_conversation(row: dict) -> str:
    conversations = row.get("conversation") or row.get("messages") or []
    parts = []
    for message in conversations:
        role = str(message.get("role") or message.get("from") or "message").title()
        content = str(message.get("content") or message.get("value") or "")
        if content:
            parts.append(f"{role}: {content}")

    return "\n".join(parts)


def is_news_dialog(text: str) -> bool:
    if has_excluded_content(text):
        return False
    score, features = news_dialog_score(text)
    has_strong_signal = "source" in features or "intent" in features
    has_supporting_signal = bool({"event", "date", "reporting"} & features)

    return score >= 4 and has_strong_signal and has_supporting_signal


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


def build_fallback_dialogs(count: int, seed: int, offset: int) -> list[NewsDialog]:
    random.seed(seed)
    dialogs = []
    templates = SAMPLE_DIALOGS
    for item_index in range(count):
        base = templates[item_index % len(templates)]
        dialog_id = f"sample-{offset + item_index + 1:03d}"
        dialogs.append(NewsDialog(dialog_id=dialog_id, source=base.source, text=base.text, created_at=base.created_at))

    return dialogs


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

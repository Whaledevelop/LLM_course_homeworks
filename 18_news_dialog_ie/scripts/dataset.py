import json
import random
from pathlib import Path
from typing import Iterable

from schemas import NewsDialog


NEWS_KEYWORDS = (
    "breaking",
    "news",
    "reported",
    "reuters",
    "ap news",
    "bbc",
    "cnn",
    "election",
    "war",
    "summit",
    "earthquake",
    "flood",
    "market",
    "minister",
    "government",
    "president",
    "company",
    "strike",
    "court",
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


def load_news_dialogs(sample_size: int, seed: int, output_path: Path) -> list[NewsDialog]:
    cached_dialogs = read_jsonl(output_path)
    if len(cached_dialogs) >= sample_size:
        return cached_dialogs[:sample_size]

    dialogs = list(stream_wildchat_news(sample_size, seed))
    if len(dialogs) < sample_size:
        dialogs.extend(build_fallback_dialogs(sample_size - len(dialogs), seed, len(dialogs)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, dialogs[:sample_size])

    return dialogs[:sample_size]


def stream_wildchat_news(sample_size: int, seed: int) -> Iterable[NewsDialog]:
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    random.seed(seed)
    dataset = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    selected = []
    for row_index, row in enumerate(dataset):
        text = flatten_conversation(row)
        if not is_news_dialog(text):
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
    lowered_text = text.lower()

    return any(keyword in lowered_text for keyword in NEWS_KEYWORDS)


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

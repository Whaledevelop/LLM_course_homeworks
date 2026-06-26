from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NewsDialog:
    dialog_id: str
    source: str
    text: str
    created_at: str = ""


@dataclass(frozen=True)
class ExtractedItem:
    label: str
    value: str
    start: int | None = None
    end: int | None = None
    confidence: float | None = None


@dataclass
class ExtractionResult:
    dialog_id: str
    extractor: str
    entities: list[ExtractedItem] = field(default_factory=list)
    events: list[ExtractedItem] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    raw_response: str = ""


@dataclass(frozen=True)
class BenchmarkResult:
    extractor: str
    examples: int
    batch_size: int
    total_seconds: float
    docs_per_second: float
    chars_per_second: float
    estimated_tokens_per_second: float
    mean_latency_ms: float
    p95_latency_ms: float
    peak_rss_mb: float
    precision: float
    recall: float
    f1: float

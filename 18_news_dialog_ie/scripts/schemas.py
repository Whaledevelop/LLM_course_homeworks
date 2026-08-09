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
    parse_valid: bool = True
    error: str = ""


@dataclass(frozen=True)
class BenchmarkResult:
    extractor: str
    model: str
    precision_mode: str
    dataset_source: str
    examples: int
    unique_examples: int
    batch_size: int
    load_seconds: float
    total_seconds: float
    docs_per_second: float
    chars_per_second: float
    estimated_tokens_per_second: float
    mean_latency_ms: float
    p95_latency_ms: float
    peak_ram_mb: float
    peak_vram_mb: float
    valid_json_rate: float
    precision: float
    recall: float
    f1: float
    micro_f1: float
    macro_f1: float
    parameter_count: int = 0
    cpu_offloaded_modules: int = 0

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PageContent:
    number: int
    text: str
    image_markdown: list[str]


@dataclass
class ConversionResult:
    pdf_path: Path
    markdown_path: Path
    images_count: int
    pages_count: int

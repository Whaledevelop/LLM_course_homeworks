from pathlib import Path

from tutor_bot.schemas.golden_dataset import GoldenDataset


def load_golden_dataset(path: Path) -> GoldenDataset:
    content = path.read_text(encoding="utf-8-sig")

    return GoldenDataset.model_validate_json(content)
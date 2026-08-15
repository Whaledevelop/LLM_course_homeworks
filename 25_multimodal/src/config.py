from dataclasses import dataclass
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    model_name: str = "laion/clap-htsat-unfused"
    dataset_name: str = "google/MusicCaps"
    data_dir: Path = PROJECT_ROOT / "data"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    sample_rate: int = 48_000
    batch_size: int = 8
    top_k: int = 5

    @property
    def metadata_dir(self) -> Path:
        return self.data_dir / "metadata"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def embeddings_dir(self) -> Path:
        return self.artifacts_dir / "embeddings"

    @property
    def indexes_dir(self) -> Path:
        return self.artifacts_dir / "indexes"

    @property
    def results_dir(self) -> Path:
        return self.artifacts_dir / "results"

    @property
    def device(self) -> torch.device:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

        return torch.device(device_name)


settings = Settings()

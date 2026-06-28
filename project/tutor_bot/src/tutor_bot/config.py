from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8-sig",
        env_prefix="TUTOR_BOT_",
        extra="ignore",
    )

    project_root: Path = Path(__file__).resolve().parents[2]
    source_notes_dir: Path
    ollama_base_url: str = "http://localhost:11434/v1"

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def notes_dir(self) -> Path:
        return self.data_dir / "notes"

    @property
    def metadata_file(self) -> Path:
        return self.data_dir / "metadata" / "notes_metadata.json"

    @property
    def history_dir(self) -> Path:
        return self.data_dir / "history"

    @property
    def indexes_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def evaluation_dir(self) -> Path:
        return self.data_dir / "evaluation"


@lru_cache
def get_settings() -> Settings:
    return Settings()

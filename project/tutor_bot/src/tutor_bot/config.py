from functools import lru_cache
from pathlib import Path

from pydantic import Field
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
    llm_provider: str = Field(default="ollama", validation_alias="LLM_PROVIDER")
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_BASE_URL",
    )
    ollama_model: str = Field(default="qwen3.5:9b", validation_alias="OLLAMA_MODEL")
    ollama_think: bool = False
    yandex_api_key: str = Field(default="", validation_alias="YANDEX_API_KEY")
    yandex_folder_id: str = Field(default="", validation_alias="YANDEX_FOLDER_ID")
    yandex_base_url: str = Field(
        default="https://ai.api.cloud.yandex.net/v1",
        validation_alias="YANDEX_BASE_URL",
    )
    yandex_model: str = Field(
        default="qwen3.6-35b-a3b",
        validation_alias="YANDEX_MODEL",
    )
    yandex_max_tokens: int = Field(default=8000, validation_alias="YANDEX_MAX_TOKENS")
    yandex_temperature: float = Field(default=0.3, validation_alias="YANDEX_TEMPERATURE")
    vllm_base_url: str = Field(
        default="http://localhost:8000/v1",
        validation_alias="VLLM_BASE_URL",
    )
    vllm_api_key: str = Field(default="", validation_alias="VLLM_API_KEY")
    vllm_model: str = Field(default="", validation_alias="VLLM_MODEL")
    langfuse_public_key: str = Field(default="", validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_base_url: str = Field(
        default="https://cloud.langfuse.com",
        validation_alias="LANGFUSE_BASE_URL",
    )

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

    @property
    def ui_state_file(self) -> Path:
        return self.data_dir / "ui_state" / "ui_state.json"

    @property
    def llm_usage_file(self) -> Path:
        return self.history_dir / "llm_usage.jsonl"


@lru_cache
def get_settings() -> Settings:
    return Settings()

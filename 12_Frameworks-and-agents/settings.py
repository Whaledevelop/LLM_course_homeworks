from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    source_file: Path
    chroma_directory: Path
    ollama_base_url: str
    chat_model: str
    embedding_model: str
    langfuse_public_key: str | None
    langfuse_secret_key: str | None
    langfuse_host: str


def get_settings() -> Settings:
    load_dotenv()
    project_directory = Path(__file__).resolve().parent

    return Settings(
        source_file=project_directory / "src" / "lesson_12_frameworks_and_agents.md",
        chroma_directory=project_directory / "data" / "chroma",
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        chat_model=os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:3b"),
        embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3"),
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        langfuse_host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )

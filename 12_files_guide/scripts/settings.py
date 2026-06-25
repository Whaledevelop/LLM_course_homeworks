from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    documents_directory: Path
    chroma_directory: Path
    hf_token: str | None
    hf_provider: str
    hf_timeout: float
    chat_model: str
    embedding_model: str
    langfuse_public_key: str | None
    langfuse_secret_key: str | None
    langfuse_host: str


def get_settings() -> Settings:
    project_directory = Path(__file__).resolve().parent.parent
    load_dotenv(project_directory / ".env")

    return Settings(
        documents_directory=project_directory / "data" / "documents",
        chroma_directory=project_directory / "data" / "chroma",
        hf_token=os.getenv("HF_TOKEN"),
        hf_provider=os.getenv("HF_PROVIDER", "auto"),
        hf_timeout=float(os.getenv("HF_TIMEOUT", "120")),
        chat_model=os.getenv("HF_CHAT_MODEL", "Qwen/Qwen3.5-9B"),
        embedding_model=os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        langfuse_host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )

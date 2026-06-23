from pathlib import Path

from dotenv import load_dotenv
import os


load_dotenv()


class Settings:
    BASE_DIR = Path(__file__).resolve().parent.parent

    DATA_DIR = BASE_DIR / "data"
    DOCUMENTS_DIR = DATA_DIR / "documents"
    CHROMA_DIR = DATA_DIR / "chroma"

    OLLAMA_BASE_URL = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    )

    OLLAMA_CHAT_MODEL = os.getenv(
        "OLLAMA_CHAT_MODEL",
        "qwen2.5:3b",
    )

    OLLAMA_EMBEDDING_MODEL = os.getenv(
        "OLLAMA_EMBEDDING_MODEL",
        "bge-m3",
    )

    @classmethod
    def ensure_directories(cls) -> None:
        cls.DOCUMENTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        cls.CHROMA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

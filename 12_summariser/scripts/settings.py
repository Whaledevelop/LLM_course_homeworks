import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


class Settings:
    BASE_DIR = Path(__file__).resolve().parent.parent
    DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "results"

    OLLAMA_BASE_URL = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    )

    OLLAMA_CHAT_MODEL = os.getenv(
        "OLLAMA_CHAT_MODEL",
        "qwen2.5:3b",
    )

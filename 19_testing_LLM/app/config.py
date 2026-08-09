import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_base_url: str
    llm_model: str
    embedding_api_key: str
    embedding_model: str
    yandex_folder_id: str
    documents_dir: Path
    chroma_dir: Path
    chunk_size: int
    chunk_overlap: int
    retrieved_chunks: int

    @classmethod
    def from_env(cls) -> "Settings":
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        openai_base_url = os.getenv("OPENAI_BASE_URL", "")
        llm_model = os.getenv("LLM_MODEL", "")
        embedding_api_key = os.getenv("EMBEDDING_API_KEY", "")
        embedding_model = os.getenv("EMBEDDING_MODEL", "")
        yandex_folder_id = os.getenv("YANDEX_FOLDER_ID", "")

        missing_variables = [
            name
            for name, value in (
                ("OPENAI_API_KEY", openai_api_key),
                ("OPENAI_BASE_URL", openai_base_url),
                ("LLM_MODEL", llm_model),
                ("EMBEDDING_API_KEY", embedding_api_key),
                ("EMBEDDING_MODEL", embedding_model),
                ("YANDEX_FOLDER_ID", yandex_folder_id),
            )
            if not value
        ]
        if missing_variables:
            names = ", ".join(missing_variables)
            raise ValueError(f"Заполните переменные окружения: {names}")

        return cls(
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            llm_model=llm_model,
            embedding_api_key=embedding_api_key,
            embedding_model=embedding_model,
            yandex_folder_id=yandex_folder_id,
            documents_dir=PROJECT_ROOT / "data" / "documents",
            chroma_dir=PROJECT_ROOT / "data" / "chroma",
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
            retrieved_chunks=int(os.getenv("RETRIEVED_CHUNKS", "3")),
        )

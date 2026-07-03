from pathlib import Path

from pydantic import BaseModel

from tutor_bot.infrastructure.atomic_json import write_json_atomically
from tutor_bot.schemas.database import (
    DatabaseIndex,
    DatabaseMetadata,
    DatabaseRegistry,
)


class DatabaseRepository:
    def __init__(self, metadata_dir: Path) -> None:
        self._metadata_dir = metadata_dir
        self._registry_file = metadata_dir / "databases.json"

    def load_registry(self) -> DatabaseRegistry:
        return self._load(self._registry_file, DatabaseRegistry())

    def save_registry(self, registry: DatabaseRegistry) -> None:
        write_json_atomically(self._registry_file, registry)

    def load_metadata(self, db_id: str) -> DatabaseMetadata:
        return self._load(self.metadata_file(db_id), DatabaseMetadata(db_id=db_id))

    def save_metadata(self, metadata: DatabaseMetadata) -> None:
        write_json_atomically(self.metadata_file(metadata.db_id), metadata)

    def load_archive(self, db_id: str) -> DatabaseMetadata:
        return self._load(self.archive_file(db_id), DatabaseMetadata(db_id=db_id))

    def save_archive(self, archive: DatabaseMetadata) -> None:
        write_json_atomically(self.archive_file(archive.db_id), archive)

    def load_index(self, root_path: Path, db_id: str) -> DatabaseIndex:
        return self._load(self.index_file(root_path), DatabaseIndex(db_id=db_id))

    def save_index(self, root_path: Path, index: DatabaseIndex) -> None:
        write_json_atomically(self.index_file(root_path), index)

    def metadata_file(self, db_id: str) -> Path:
        return self._metadata_dir / f"{db_id}_metadata.json"

    def archive_file(self, db_id: str) -> Path:
        return self._metadata_dir / f"{db_id}_metadata_archive.json"

    @staticmethod
    def index_file(root_path: Path) -> Path:
        return root_path / "tutor_bot_db_index_data.json"

    def _load(self, path: Path, default: BaseModel):
        if not path.is_file():
            return default

        return type(default).model_validate_json(path.read_text(encoding="utf-8-sig"))

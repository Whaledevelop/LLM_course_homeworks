import os
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from tutor_bot.schemas.notes_metadata_catalog import NotesMetadataCatalog


class NotesMetadataRepository:
    def __init__(self, metadata_file: Path) -> None:
        self._metadata_file = metadata_file
        self._backup_dir = metadata_file.parent / "backups"

    def load(self) -> NotesMetadataCatalog:
        content = self._metadata_file.read_text(encoding="utf-8-sig")

        return NotesMetadataCatalog.model_validate_json(content)

    def save(
        self,
        catalog: NotesMetadataCatalog,
    ) -> NotesMetadataCatalog:
        version_time = datetime.now().astimezone()
        updated_catalog = catalog.model_copy(update={"version_time": version_time})

        serialized_catalog = updated_catalog.model_dump_json(indent=2) + "\n"
        temporary_file = self._metadata_file.with_name(
            f".{self._metadata_file.name}.{uuid4().hex}.tmp"
        )

        try:
            with temporary_file.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as file:
                file.write(serialized_catalog)
                file.flush()
                os.fsync(file.fileno())

            self._create_backup(version_time)
            temporary_file.replace(self._metadata_file)
        finally:
            temporary_file.unlink(missing_ok=True)

        return updated_catalog

    def _create_backup(
        self,
        version_time: datetime,
    ) -> Path:
        self._backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = version_time.strftime("%Y%m%dT%H%M%S%f%z")
        backup_file = self._backup_dir / f"notes_metadata-{timestamp}.json"

        shutil.copy2(
            self._metadata_file,
            backup_file,
        )

        return backup_file

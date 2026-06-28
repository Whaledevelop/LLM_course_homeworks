from pathlib import Path

from tutor_bot.schemas.notes_metadata_catalog import NotesMetadataCatalog


class NotesMetadataRepository:
    def __init__(self, metadata_file: Path) -> None:
        self._metadata_file = metadata_file

    def load(self) -> NotesMetadataCatalog:
        content = self._metadata_file.read_text(encoding="utf-8-sig")

        return NotesMetadataCatalog.model_validate_json(content)

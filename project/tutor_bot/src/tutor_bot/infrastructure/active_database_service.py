import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tutor_bot.infrastructure.database_indexing_service import (
    DatabaseIndexingResult,
    DatabaseIndexingService,
)
from tutor_bot.infrastructure.database_repository import DatabaseRepository
from tutor_bot.schemas.database import (
    DatabaseMetadata,
    DatabaseNoteMetadata,
    DatabaseRegistration,
)


@dataclass(frozen=True, slots=True)
class ActiveDatabaseContext:
    db_id: str
    root_path: Path
    metadata_file: Path
    index_file: Path
    indexes_dir: Path


@dataclass(frozen=True, slots=True)
class DatabaseSummary:
    db_id: str
    root_path: Path
    note_count: int
    archived_note_count: int
    groups: dict[str, int]
    is_active: bool


class ActiveDatabaseService:
    def __init__(self, project_data_dir: Path) -> None:
        self._data_dir = project_data_dir
        self._repository = DatabaseRepository(project_data_dir / "metadata")
        self._indexing_service = DatabaseIndexingService(self._repository)

    def list_databases(self) -> tuple[str, ...]:
        return tuple(sorted(self._repository.load_registry().databases))

    def list_summaries(self) -> tuple[DatabaseSummary, ...]:
        registry = self._repository.load_registry()
        summaries = []

        for db_id, registration in sorted(registry.databases.items()):
            metadata = self._repository.load_metadata(db_id)
            archive = self._repository.load_archive(db_id)
            groups: dict[str, int] = {}

            for note_metadata in metadata.notes.values():
                group = note_metadata.group or "Без группы"
                groups[group] = groups.get(group, 0) + 1

            summaries.append(
                DatabaseSummary(
                    db_id=db_id,
                    root_path=registration.root_path,
                    note_count=len(metadata.notes),
                    archived_note_count=len(archive.notes),
                    groups=groups,
                    is_active=db_id == registry.active_db_id,
                )
            )

        return tuple(summaries)

    def get_active(self) -> ActiveDatabaseContext | None:
        registry = self._repository.load_registry()

        if registry.active_db_id is None:
            return None

        registration = registry.databases.get(registry.active_db_id)

        if registration is None:
            raise ValueError(f"Active DB is not registered: {registry.active_db_id}")

        return self._create_context(registry.active_db_id, registration.root_path)

    def register(self, db_id: str, root_path: Path) -> DatabaseIndexingResult:
        root_path = root_path.resolve()
        registry = self._repository.load_registry()

        if db_id in registry.databases:
            raise ValueError(f"DB is already registered: {db_id}")

        result = self._indexing_service.update(db_id, root_path)
        self._migrate_legacy_metadata(db_id)
        databases = dict(registry.databases)
        databases[db_id] = DatabaseRegistration(root_path=root_path)
        updated_registry = registry.model_copy(
            update={"active_db_id": db_id, "databases": databases}
        )
        self._repository.save_registry(updated_registry)

        return result

    def select(self, db_id: str) -> ActiveDatabaseContext:
        registry = self._repository.load_registry()

        if db_id not in registry.databases:
            raise KeyError(f"DB is not registered: {db_id}")

        self._repository.save_registry(registry.model_copy(update={"active_db_id": db_id}))

        return self._create_context(db_id, registry.databases[db_id].root_path)

    def update_active(self) -> DatabaseIndexingResult:
        context = self.get_active()

        if context is None:
            raise ValueError("Active DB is not selected")

        return self._indexing_service.update(context.db_id, context.root_path)

    def remove(self, db_id: str) -> None:
        registry = self._repository.load_registry()
        registration = registry.databases.get(db_id)

        if registration is None:
            raise KeyError(f"DB is not registered: {db_id}")

        databases = dict(registry.databases)
        del databases[db_id]
        active_db_id = registry.active_db_id

        if active_db_id == db_id:
            active_db_id = sorted(databases)[0] if databases else None

        self._repository.save_registry(
            registry.model_copy(update={"active_db_id": active_db_id, "databases": databases})
        )
        self._repository.metadata_file(db_id).unlink(missing_ok=True)
        self._repository.archive_file(db_id).unlink(missing_ok=True)
        self._repository.index_file(registration.root_path).unlink(missing_ok=True)
        shutil.rmtree(self._data_dir / "indexes" / db_id, ignore_errors=True)
        shutil.rmtree(self._data_dir / "history" / db_id, ignore_errors=True)
        (self._data_dir / "ui_state" / f"{db_id}.json").unlink(missing_ok=True)

    def _create_context(self, db_id: str, root_path: Path) -> ActiveDatabaseContext:
        return ActiveDatabaseContext(
            db_id=db_id,
            root_path=root_path,
            metadata_file=self._repository.metadata_file(db_id),
            index_file=self._repository.index_file(root_path),
            indexes_dir=self._data_dir / "indexes" / db_id,
        )

    def _migrate_legacy_metadata(self, db_id: str) -> None:
        legacy_file = self._data_dir / "metadata" / "notes_metadata.json"

        if not legacy_file.is_file():
            return

        metadata = self._repository.load_metadata(db_id)
        legacy_content = json.loads(legacy_file.read_text(encoding="utf-8-sig"))
        legacy_notes = legacy_content.get("notes", {})
        migrated_notes = dict(metadata.notes)

        for note_id in metadata.notes:
            legacy_note = legacy_notes.get(str(note_id))

            if legacy_note is None:
                continue

            migrated_notes[note_id] = DatabaseNoteMetadata(
                group=str(legacy_note.get("group", legacy_note.get("theme", ""))),
                comment=str(legacy_note.get("comment", "")),
                importance=int(legacy_note.get("importance", 0)),
                knowledge=int(legacy_note.get("knowledge", 0)),
            )

        backup_dir = legacy_file.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f%z")
        backup_file = backup_dir / f"notes_metadata-before-multidb-{timestamp}.json"
        shutil.copy2(legacy_file, backup_file)
        self._repository.save_metadata(DatabaseMetadata(db_id=db_id, notes=migrated_notes))

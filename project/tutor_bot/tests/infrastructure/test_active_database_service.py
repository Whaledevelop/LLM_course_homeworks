import json
from pathlib import Path
from uuid import UUID

from tutor_bot.infrastructure.active_database_service import ActiveDatabaseService
from tutor_bot.infrastructure.database_repository import DatabaseRepository
from tutor_bot.application.create_note_command import CreateNoteCommand
from tutor_bot.infrastructure.database_notes_repository import DatabaseNotesRepository
from tutor_bot.infrastructure.file_note_command_service import FileNoteCommandService
from tutor_bot.infrastructure.metadata_note_query_service import MetadataNoteQueryService


_NOTE_ID = UUID("0b2c61a8-505a-552c-b731-7b9627970eff")


def test_registers_selects_and_isolates_databases(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    unity_root = tmp_path / "unity"
    llm_root = tmp_path / "llm"
    unity_root.mkdir()
    llm_root.mkdir()
    (unity_root / "unity.md").write_text("# Unity\n")
    (llm_root / "llm.md").write_text("# LLM\n")
    service = ActiveDatabaseService(data_dir)

    service.register("Unity", unity_root)
    service.register("LLM", llm_root)
    service.select("Unity")

    repository = DatabaseRepository(data_dir / "metadata")
    unity_index = repository.load_index(unity_root, "Unity")
    llm_index = repository.load_index(llm_root, "LLM")
    assert service.get_active().db_id == "Unity"
    assert set(unity_index.notes).isdisjoint(llm_index.notes)
    assert service.list_databases() == ("LLM", "Unity")
    summaries = {summary.db_id: summary for summary in service.list_summaries()}
    assert summaries["Unity"].note_count == 1
    assert summaries["Unity"].is_active
    assert summaries["LLM"].note_count == 1
    assert not summaries["LLM"].is_active


def test_migrates_legacy_metadata_with_backup(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    notes_root = tmp_path / "notes"
    metadata_dir = data_dir / "metadata"
    notes_root.mkdir()
    metadata_dir.mkdir(parents=True)
    (notes_root / "note.md").write_text(
        f"---\nid: {_NOTE_ID}\n---\n# Note\n",
        encoding="utf-8",
    )
    legacy_metadata = {
        "version": 2,
        "version_time": "2026-07-03T00:00:00+03:00",
        "notes": {
            str(_NOTE_ID): {
                "theme": "unity",
                "comment": "repeat",
                "importance": 8,
                "knowledge": 3,
                "last_recorded_name": "Note",
                "relative_path": "note.md",
            }
        },
    }
    legacy_file = metadata_dir / "notes_metadata.json"
    legacy_file.write_text(json.dumps(legacy_metadata), encoding="utf-8")

    ActiveDatabaseService(data_dir).register("Unity", notes_root)

    metadata = DatabaseRepository(metadata_dir).load_metadata("Unity")
    migrated_note = metadata.notes[_NOTE_ID]
    backups = tuple((metadata_dir / "backups").glob("notes_metadata-before-multidb-*.json"))
    assert migrated_note.group == "unity"
    assert migrated_note.comment == "repeat"
    assert migrated_note.importance == 8
    assert migrated_note.knowledge == 3
    assert len(backups) == 1
    assert legacy_file.is_file()


def test_crud_uses_only_selected_database_files(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    unity_root = tmp_path / "unity"
    llm_root = tmp_path / "llm"
    unity_root.mkdir()
    llm_root.mkdir()
    database_service = ActiveDatabaseService(data_dir)
    database_service.register("Unity", unity_root)
    database_service.register("LLM", llm_root)
    repository = DatabaseNotesRepository(data_dir / "metadata", "Unity", unity_root)
    command_service = FileNoteCommandService(repository, unity_root)
    query_service = MetadataNoteQueryService(repository, unity_root)

    created_note = command_service.create_note(
        CreateNoteCommand(
            title="SOLID",
            markdown_content="Principles",
            group="architecture",
            comment="repeat",
            importance=8,
            knowledge=2,
        )
    )

    loaded_note = query_service.get_note(created_note.id)
    assert loaded_note.title == str(created_note.id)
    assert loaded_note.group == "architecture"
    assert tuple(unity_root.rglob("*.md"))
    assert not tuple(llm_root.rglob("*.md"))
    assert (
        created_note.id
        in DatabaseRepository(data_dir / "metadata")
        .load_index(
            unity_root,
            "Unity",
        )
        .notes
    )


def test_removes_database_artifacts_without_deleting_notes(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    notes_root = tmp_path / "notes"
    notes_root.mkdir()
    note_path = notes_root / "note.md"
    note_path.write_text("# Note\n", encoding="utf-8")
    service = ActiveDatabaseService(data_dir)
    service.register("Unity", notes_root)
    indexed_note_content = note_path.read_text(encoding="utf-8")
    indexes_dir = data_dir / "indexes" / "Unity"
    indexes_dir.mkdir(parents=True)
    (indexes_dir / "artifact").write_text("index", encoding="utf-8")

    service.remove("Unity")

    assert note_path.is_file()
    assert note_path.read_text(encoding="utf-8") == indexed_note_content
    assert not (notes_root / "tutor_bot_db_index_data.json").exists()
    assert not (data_dir / "metadata" / "Unity_metadata.json").exists()
    assert not (data_dir / "metadata" / "Unity_metadata_archive.json").exists()
    assert not indexes_dir.exists()
    assert service.list_databases() == ()
    assert service.get_active() is None

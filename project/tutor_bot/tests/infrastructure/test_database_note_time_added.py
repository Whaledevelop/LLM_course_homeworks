from uuid import uuid4

from tutor_bot.infrastructure.database_indexing_service import DatabaseIndexingService
from tutor_bot.infrastructure.database_repository import DatabaseRepository


def test_initial_database_notes_share_time_added_and_keep_it_on_update(tmp_path):
    metadata_dir = tmp_path / "metadata"
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    first_note_id = uuid4()
    second_note_id = uuid4()
    (notes_dir / "first.md").write_text(
        f"---\ntutor_bot_note_id: {first_note_id}\n---\nFirst",
        encoding="utf-8",
    )
    (notes_dir / "second.md").write_text(
        f"---\ntutor_bot_note_id: {second_note_id}\n---\nSecond",
        encoding="utf-8",
    )
    repository = DatabaseRepository(metadata_dir)
    indexing_service = DatabaseIndexingService(repository)

    indexing_service.update("notes", notes_dir)
    initial_metadata = repository.load_metadata("notes")
    initial_time_added = initial_metadata.notes[first_note_id].time_added

    assert initial_metadata.notes[second_note_id].time_added == initial_time_added
    assert initial_time_added.tzinfo is not None

    indexing_service.update("notes", notes_dir)
    updated_metadata = repository.load_metadata("notes")

    assert updated_metadata.notes[first_note_id].time_added == initial_time_added
    assert updated_metadata.notes[second_note_id].time_added == initial_time_added

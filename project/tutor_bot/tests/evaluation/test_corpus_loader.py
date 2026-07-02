from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from uuid import uuid4

from tutor_bot.evaluation.corpus_loader import resolve_corpus_files
from tutor_bot.schemas.evaluation_corpus import EvaluationCorpus
from tutor_bot.schemas.note_metadata import NoteMetadata
from tutor_bot.schemas.notes_metadata_catalog import NotesMetadataCatalog


def test_resolves_corpus_files(tmp_path: Path) -> None:
    source_notes_dir = tmp_path / "notes"
    topic_dir = source_notes_dir / "topic"
    topic_dir.mkdir(parents=True)

    note_ids = []
    metadata_entries = {}

    for i in range(15):
        note_id = uuid4()
        relative_path = PurePosixPath("topic") / f"note-{i}.md"
        note_path = source_notes_dir / relative_path

        note_path.write_text(
            f"---\nid: {note_id}\n---\n\nTest content",
            encoding="utf-8",
        )

        note_ids.append(note_id)
        metadata_entries[note_id] = NoteMetadata(
            theme="topic",
            comment="",
            importance=5,
            knowledge=0,
            last_recorded_name=f"note-{i}",
            relative_path=relative_path,
        )

    corpus = EvaluationCorpus(
        version=1,
        note_ids=note_ids,
    )

    metadata_catalog = NotesMetadataCatalog(
        version=2,
        version_time=datetime.now(timezone.utc),
        notes=metadata_entries,
    )

    resolved_files = resolve_corpus_files(
        corpus,
        metadata_catalog,
        source_notes_dir,
    )

    assert len(resolved_files) == 15
    assert all(path.is_file() for path in resolved_files.values())

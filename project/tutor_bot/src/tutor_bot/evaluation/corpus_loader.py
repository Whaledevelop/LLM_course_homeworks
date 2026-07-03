import re
from pathlib import Path
from uuid import UUID

from tutor_bot.schemas.evaluation_corpus import EvaluationCorpus
from tutor_bot.schemas.notes_metadata_catalog import NotesMetadataCatalog


_FRONTMATTER_PATTERN = re.compile(
    r"\A---\s*\n(?P<content>.*?)\n---(?:\n|\Z)",
    re.DOTALL,
)

_NOTE_ID_PATTERN = re.compile(
    r"^tutor_bot_note_id:\s*(?P<note_id>[^\s]+)\s*$",
    re.MULTILINE,
)


def load_evaluation_corpus(path: Path) -> EvaluationCorpus:
    content = path.read_text(encoding="utf-8-sig")

    return EvaluationCorpus.model_validate_json(content)


def load_notes_metadata(path: Path) -> NotesMetadataCatalog:
    content = path.read_text(encoding="utf-8-sig")

    return NotesMetadataCatalog.model_validate_json(content)


def resolve_corpus_files(
    corpus: EvaluationCorpus,
    metadata_catalog: NotesMetadataCatalog,
    source_notes_dir: Path,
) -> dict[UUID, Path]:
    missing_note_ids = [
        note_id for note_id in corpus.note_ids if note_id not in metadata_catalog.notes
    ]

    if missing_note_ids:
        formatted_ids = ", ".join(str(note_id) for note_id in missing_note_ids)
        raise ValueError(f"Corpus note ids missing from metadata: {formatted_ids}")

    source_root = source_notes_dir.resolve()
    resolved_files: dict[UUID, Path] = {}

    for note_id in corpus.note_ids:
        metadata = metadata_catalog.notes[note_id]
        note_path = (source_root / metadata.relative_path).resolve()

        if not note_path.is_relative_to(source_root):
            raise ValueError(f"Note path escapes source directory: {note_path}")

        if not note_path.is_file():
            raise FileNotFoundError(f"Corpus note not found: {note_path}")

        actual_note_id = _read_note_id(note_path)

        if actual_note_id != note_id:
            raise ValueError(
                f"Note id mismatch for {note_path}: expected {note_id}, found {actual_note_id}"
            )

        resolved_files[note_id] = note_path

    return resolved_files


def _read_note_id(path: Path) -> UUID:
    content = path.read_text(encoding="utf-8-sig")
    frontmatter_match = _FRONTMATTER_PATTERN.match(content)

    if frontmatter_match is None:
        raise ValueError(f"Markdown frontmatter not found: {path}")

    note_id_match = _NOTE_ID_PATTERN.search(frontmatter_match.group("content"))

    if note_id_match is None:
        raise ValueError(f"Note id not found in frontmatter: {path}")

    return UUID(note_id_match.group("note_id"))

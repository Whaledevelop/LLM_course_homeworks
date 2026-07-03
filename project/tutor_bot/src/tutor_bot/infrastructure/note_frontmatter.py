import os
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4


_FRONTMATTER_PATTERN = re.compile(
    r"\A---[ \t]*\r?\n(?P<content>.*?)\r?\n---[ \t]*(?:\r?\n|\Z)",
    re.DOTALL,
)
_NOTE_ID_PATTERN = re.compile(
    r"^(?P<key>tutor_bot_note_id|id):[ \t]*(?P<value>[^\r\n#]+?)[ \t]*$",
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class PreparedNote:
    note_id: UUID
    original_content: str
    updated_content: str

    @property
    def changed(self) -> bool:
        return self.original_content != self.updated_content


def prepare_note(path: Path) -> PreparedNote:
    original_content = path.read_text(encoding="utf-8-sig")
    frontmatter_match = _FRONTMATTER_PATTERN.match(original_content)

    if original_content.startswith("---") and frontmatter_match is None:
        raise ValueError(f"Malformed Markdown frontmatter: {path}")

    if frontmatter_match is None:
        note_id = uuid4()
        separator = "" if original_content.startswith(("\n", "\r")) else "\n"
        updated_content = f"---\ntutor_bot_note_id: {note_id}\n---\n{separator}{original_content}"

        return PreparedNote(note_id, original_content, updated_content)

    frontmatter = frontmatter_match.group("content")
    matches = list(_NOTE_ID_PATTERN.finditer(frontmatter))
    tutor_matches = [match for match in matches if match.group("key") == "tutor_bot_note_id"]
    legacy_matches = [match for match in matches if match.group("key") == "id"]

    if len(tutor_matches) > 1 or len(legacy_matches) > 1:
        raise ValueError(f"Duplicate note id field in frontmatter: {path}")

    selected_match = (
        tutor_matches[0] if tutor_matches else legacy_matches[0] if legacy_matches else None
    )
    note_id = UUID(selected_match.group("value").strip()) if selected_match else uuid4()
    updated_frontmatter = f"tutor_bot_note_id: {note_id}"

    updated_content = (
        original_content[: frontmatter_match.start("content")]
        + updated_frontmatter
        + original_content[frontmatter_match.end("content") :]
    )

    return PreparedNote(note_id, original_content, updated_content)


def write_note_atomically(path: Path, content: str) -> None:
    temporary_file = path.with_name(f".{path.name}.{uuid4().hex}.tmp")

    try:
        with temporary_file.open("w", encoding="utf-8", newline="") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())

        temporary_file.replace(path)
    finally:
        temporary_file.unlink(missing_ok=True)

import re
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID


_FRONTMATTER_PATTERN = re.compile(
    r"\A---\s*\n(?P<content>.*?)\n---(?:\n|\Z)",
    re.DOTALL,
)

_NOTE_ID_PATTERN = re.compile(
    r"^tutor_bot_note_id:\s*(?P<note_id>[^\s]+)\s*$",
    re.MULTILINE,
)


@dataclass(
    frozen=True,
    slots=True,
)
class MarkdownDocument:
    note_id: UUID
    content: str

    @property
    def normalized_content(self) -> str:
        return self.content.rstrip() + "\n"

    def serialize(self) -> str:
        return f"---\ntutor_bot_note_id: {self.note_id}\n---\n\n{self.normalized_content}"

    @classmethod
    def parse(
        cls,
        file_content: str,
        note_path: Path,
    ) -> "MarkdownDocument":
        frontmatter_match = _FRONTMATTER_PATTERN.match(file_content)

        if frontmatter_match is None:
            raise ValueError(f"Markdown frontmatter not found: {note_path}")

        note_id_match = _NOTE_ID_PATTERN.search(frontmatter_match.group("content"))

        if note_id_match is None:
            raise ValueError(f"Note id not found in frontmatter: {note_path}")

        note_id = UUID(note_id_match.group("note_id"))
        content = file_content[frontmatter_match.end() :].lstrip("\r\n")

        return cls(
            note_id=note_id,
            content=content,
        )

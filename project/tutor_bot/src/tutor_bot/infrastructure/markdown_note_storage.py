import os
from pathlib import Path, PurePosixPath
from uuid import UUID, uuid4

from tutor_bot.infrastructure.markdown_document import MarkdownDocument


_CREATED_NOTES_DIRECTORY = PurePosixPath("_tutor_bot")


class MarkdownNoteStorage:
    def __init__(
        self,
        source_notes_dir: Path,
    ) -> None:
        self._source_root = source_notes_dir.resolve()

    def create(
        self,
        note_id: UUID,
        title: str,
        markdown_content: str,
    ) -> tuple[PurePosixPath, MarkdownDocument]:
        relative_path = _CREATED_NOTES_DIRECTORY / f"{note_id}.md"
        note_path = self._resolve_path(relative_path)

        if note_path.exists():
            raise FileExistsError(f"Note file already exists: {note_path}")

        note_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        document = MarkdownDocument(
            note_id=note_id,
            content=markdown_content,
            title=title,
        )

        self._write_atomically(
            note_path,
            document.serialize(),
        )

        return relative_path, document

    def update(
        self,
        note_id: UUID,
        relative_path: PurePosixPath,
        title: str,
        markdown_content: str,
    ) -> tuple[str, MarkdownDocument]:
        note_path = self._get_existing_path(relative_path)
        original_file_content, _ = self._read_document(
            note_id,
            note_path,
        )

        updated_document = MarkdownDocument(
            note_id=note_id,
            content=markdown_content,
            title=title,
        )

        self._write_atomically(
            note_path,
            updated_document.serialize(),
        )

        return original_file_content, updated_document

    def restore(
        self,
        relative_path: PurePosixPath,
        file_content: str,
    ) -> None:
        note_path = self._resolve_path(relative_path)

        self._write_atomically(
            note_path,
            file_content,
        )

    def delete_created(
        self,
        relative_path: PurePosixPath,
    ) -> None:
        note_path = self._resolve_path(relative_path)
        note_path.unlink(missing_ok=True)

    def stage_delete(
        self,
        note_id: UUID,
        relative_path: PurePosixPath,
    ) -> tuple[Path, MarkdownDocument]:
        note_path = self._get_existing_path(relative_path)
        _, document = self._read_document(
            note_id,
            note_path,
        )

        staged_file = note_path.with_name(f".{note_path.name}.{uuid4().hex}.deleted")

        note_path.replace(staged_file)

        return staged_file, document

    def restore_deleted(
        self,
        staged_file: Path,
        relative_path: PurePosixPath,
    ) -> None:
        note_path = self._resolve_path(relative_path)
        staged_file.replace(note_path)

    def finalize_deleted(
        self,
        staged_file: Path,
    ) -> None:
        staged_file.unlink()

    def _read_document(
        self,
        expected_note_id: UUID,
        note_path: Path,
    ) -> tuple[str, MarkdownDocument]:
        file_content = note_path.read_text(encoding="utf-8-sig")
        document = MarkdownDocument.parse(
            file_content,
            note_path,
        )

        if document.note_id != expected_note_id:
            raise ValueError(
                f"Note id mismatch: expected {expected_note_id}, found {document.note_id}"
            )

        return file_content, document

    def _get_existing_path(
        self,
        relative_path: PurePosixPath,
    ) -> Path:
        note_path = self._resolve_path(relative_path)

        if not note_path.is_file():
            raise FileNotFoundError(f"Note file not found: {note_path}")

        return note_path

    def _resolve_path(
        self,
        relative_path: PurePosixPath,
    ) -> Path:
        note_path = (self._source_root / relative_path).resolve()

        if not note_path.is_relative_to(self._source_root):
            raise ValueError(f"Note path escapes source directory: {note_path}")

        return note_path

    def _write_atomically(
        self,
        path: Path,
        content: str,
    ) -> None:
        temporary_file = path.with_name(f".{path.name}.{uuid4().hex}.tmp")

        try:
            with temporary_file.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())

            temporary_file.replace(path)
        finally:
            temporary_file.unlink(missing_ok=True)

from collections import defaultdict
from pathlib import Path, PurePosixPath
from uuid import UUID

from tutor_bot.application.note_consistency_report import (
    NoteConsistencyReport,
)
from tutor_bot.infrastructure.markdown_document import MarkdownDocument
from tutor_bot.infrastructure.notes_metadata_repository import (
    NotesMetadataRepository,
)


class NoteConsistencyChecker:
    def __init__(
        self,
        metadata_repository: NotesMetadataRepository,
        source_notes_dir: Path,
    ) -> None:
        self._metadata_repository = metadata_repository
        self._source_notes_dir = source_notes_dir.resolve()

    def check(self) -> NoteConsistencyReport:
        catalog = self._metadata_repository.load()
        markdown_paths = self._find_markdown_paths()

        metadata_ids_by_path: dict[str, list[UUID]] = defaultdict(list)
        metadata_paths_by_normalized_path: dict[
            str,
            list[PurePosixPath],
        ] = defaultdict(list)

        for note_id, metadata in catalog.notes.items():
            normalized_path = self._normalize_path(metadata.relative_path)
            metadata_ids_by_path[normalized_path].append(note_id)
            metadata_paths_by_normalized_path[normalized_path].append(metadata.relative_path)

        markdown_ids_by_note_id: dict[
            UUID,
            list[PurePosixPath],
        ] = defaultdict(list)

        parsed_documents: dict[
            PurePosixPath,
            MarkdownDocument,
        ] = {}

        invalid_markdown_paths = []

        for relative_path, note_path in markdown_paths.items():
            try:
                file_content = note_path.read_text(encoding="utf-8-sig")
                document = MarkdownDocument.parse(
                    file_content,
                    note_path,
                )
            except (UnicodeError, ValueError):
                invalid_markdown_paths.append(relative_path)
                continue

            parsed_documents[relative_path] = document
            markdown_ids_by_note_id[document.note_id].append(relative_path)

        normalized_markdown_paths = {
            self._normalize_path(relative_path) for relative_path in markdown_paths
        }

        missing_markdown_note_ids = [
            note_id
            for normalized_path, note_ids in metadata_ids_by_path.items()
            if normalized_path not in normalized_markdown_paths
            for note_id in note_ids
        ]

        markdown_without_metadata = [
            relative_path
            for relative_path, document in parsed_documents.items()
            if document.note_id not in catalog.notes
        ]

        duplicate_markdown_note_ids = [
            note_id
            for note_id, relative_paths in markdown_ids_by_note_id.items()
            if len(relative_paths) > 1
        ]

        duplicate_metadata_paths = [
            relative_paths[0]
            for relative_paths in metadata_paths_by_normalized_path.values()
            if len(relative_paths) > 1
        ]

        frontmatter_id_mismatches = []

        for relative_path, document in parsed_documents.items():
            normalized_path = self._normalize_path(relative_path)
            expected_note_ids = metadata_ids_by_path.get(normalized_path)

            if not expected_note_ids:
                continue

            if document.note_id not in expected_note_ids:
                frontmatter_id_mismatches.append(relative_path)

        return NoteConsistencyReport(
            missing_markdown_note_ids=tuple(
                sorted(
                    missing_markdown_note_ids,
                    key=str,
                )
            ),
            markdown_without_metadata=self._sort_paths(markdown_without_metadata),
            duplicate_markdown_note_ids=tuple(
                sorted(
                    duplicate_markdown_note_ids,
                    key=str,
                )
            ),
            duplicate_metadata_paths=self._sort_paths(duplicate_metadata_paths),
            frontmatter_id_mismatches=self._sort_paths(frontmatter_id_mismatches),
            invalid_markdown_paths=self._sort_paths(invalid_markdown_paths),
        )

    def _find_markdown_paths(self) -> dict[PurePosixPath, Path]:
        markdown_paths = {}

        for note_path in self._source_notes_dir.rglob("*.md"):
            relative_path = PurePosixPath(note_path.relative_to(self._source_notes_dir).as_posix())
            markdown_paths[relative_path] = note_path

        return markdown_paths

    def _normalize_path(
        self,
        relative_path: PurePosixPath,
    ) -> str:
        return relative_path.as_posix().casefold()

    def _sort_paths(
        self,
        paths: list[PurePosixPath],
    ) -> tuple[PurePosixPath, ...]:
        return tuple(
            sorted(
                paths,
                key=self._normalize_path,
            )
        )

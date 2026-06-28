from tutor_bot.application.note_list_item import NoteListItem
from tutor_bot.infrastructure.notes_metadata_repository import (
    NotesMetadataRepository,
)


class MetadataNoteQueryService:
    def __init__(
        self,
        metadata_repository: NotesMetadataRepository,
    ) -> None:
        self._metadata_repository = metadata_repository

    def list_notes(self) -> list[NoteListItem]:
        metadata_catalog = self._metadata_repository.load()

        note_items = [
            NoteListItem(
                id=note_id,
                title=metadata.last_recorded_name,
                theme=metadata.theme,
                difficulty=metadata.difficulty,
                importance=metadata.importance,
                completeness=metadata.completeness,
                mastery=metadata.mastery,
            )
            for note_id, metadata in metadata_catalog.notes.items()
        ]

        return sorted(
            note_items,
            key=lambda note_item: note_item.title.casefold(),
        )

from typing import Protocol
from uuid import UUID

from tutor_bot.application.note_details import NoteDetails
from tutor_bot.application.note_list_item import NoteListItem


class NoteQueryService(Protocol):
    def list_notes(self) -> list[NoteListItem]: ...

    def get_note(self, note_id: UUID) -> NoteDetails: ...

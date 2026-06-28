from typing import Protocol

from tutor_bot.application.note_list_item import NoteListItem


class NoteQueryService(Protocol):
    def list_notes(self) -> list[NoteListItem]: ...

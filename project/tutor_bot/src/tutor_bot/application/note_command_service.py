from typing import Protocol

from tutor_bot.application.note_details import NoteDetails
from tutor_bot.application.update_note_command import UpdateNoteCommand


class NoteCommandService(Protocol):
    def update_note(
        self,
        command: UpdateNoteCommand,
    ) -> NoteDetails: ...

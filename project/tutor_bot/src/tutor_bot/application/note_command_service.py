from typing import Protocol

from tutor_bot.application.create_note_command import CreateNoteCommand
from tutor_bot.application.delete_note_command import DeleteNoteCommand
from tutor_bot.application.note_details import NoteDetails
from tutor_bot.application.update_note_command import UpdateNoteCommand


class NoteCommandService(Protocol):
    def create_note(
        self,
        command: CreateNoteCommand,
    ) -> NoteDetails: ...

    def update_note(
        self,
        command: UpdateNoteCommand,
    ) -> NoteDetails: ...

    def delete_note(
        self,
        command: DeleteNoteCommand,
    ) -> NoteDetails: ...

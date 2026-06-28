from pydantic import Field

from tutor_bot.application.note_list_item import NoteListItem


class NoteDetails(NoteListItem):
    comment: str
    markdown_content: str = Field(min_length=1)

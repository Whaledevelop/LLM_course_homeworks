from tutor_bot.application.note_list_item import NoteListItem


class NoteDetails(NoteListItem):
    comment: str
    questions_for_tests: tuple[str, ...] = ()
    markdown_content: str

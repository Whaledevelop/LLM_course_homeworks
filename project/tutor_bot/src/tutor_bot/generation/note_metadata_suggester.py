from typing import Protocol

from tutor_bot.application.note_metadata_suggestion import NoteMetadataSuggestion


class NoteMetadataSuggester(Protocol):
    def suggest(
        self,
        markdown_content: str,
    ) -> NoteMetadataSuggestion: ...

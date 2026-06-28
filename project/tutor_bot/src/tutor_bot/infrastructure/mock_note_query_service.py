from uuid import UUID

from tutor_bot.application.note_list_item import NoteListItem


class MockNoteQueryService:
    def list_notes(self) -> list[NoteListItem]:
        return [
            NoteListItem(
                id=UUID("e248dd6f-963d-552f-8616-d87082905b4e"),
                title="Поточность, синхронность, параллелизм",
                theme="threads",
                difficulty="middle",
                importance=9,
                completeness=9,
                mastery=2,
            ),
            NoteListItem(
                id=UUID("2e2a0b1a-43f0-5d43-918f-393d557d5eac"),
                title="Garbage collector",
                theme="csharp",
                difficulty="middle",
                importance=8,
                completeness=7,
                mastery=1,
            ),
            NoteListItem(
                id=UUID("00ddf68b-1408-5075-813e-9f897e8ee27a"),
                title="SOLID",
                theme="architecture",
                difficulty="middle",
                importance=8,
                completeness=3,
                mastery=1,
            ),
        ]

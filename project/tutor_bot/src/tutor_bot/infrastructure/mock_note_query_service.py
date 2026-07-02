from uuid import UUID

from tutor_bot.application.note_list_item import NoteListItem


class MockNoteQueryService:
    def list_notes(self) -> list[NoteListItem]:
        return [
            NoteListItem(
                id=UUID("e248dd6f-963d-552f-8616-d87082905b4e"),
                title="Поточность, синхронность, параллелизм",
                theme="threads",
                importance=9,
                knowledge=2,
            ),
            NoteListItem(
                id=UUID("2e2a0b1a-43f0-5d43-918f-393d557d5eac"),
                title="Garbage collector",
                theme="csharp",
                importance=8,
                knowledge=1,
            ),
            NoteListItem(
                id=UUID("00ddf68b-1408-5075-813e-9f897e8ee27a"),
                title="SOLID",
                theme="architecture",
                importance=8,
                knowledge=1,
            ),
        ]

from typing import Protocol


class NoteContentGenerator(Protocol):
    def generate(
        self,
        title: str,
        existing_content: str = "",
        fullness: int = 7,
    ) -> str:
        pass

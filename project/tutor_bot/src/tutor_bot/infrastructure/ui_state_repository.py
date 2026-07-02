import json
from pathlib import Path
from uuid import UUID


class UiStateRepository:
    def __init__(
        self,
        state_file: Path,
    ) -> None:
        self._state_file = state_file

    def load_selected_note_id(self) -> UUID | None:
        if not self._state_file.is_file():
            return None

        content = json.loads(self._state_file.read_text(encoding="utf-8-sig"))
        selected_note_id = content.get("selected_note_id")

        if not selected_note_id:
            return None

        return UUID(str(selected_note_id))

    def save_selected_note_id(
        self,
        note_id: UUID,
    ) -> None:
        self._state_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        content = {
            "selected_note_id": str(note_id),
        }
        serialized_content = json.dumps(
            content,
            ensure_ascii=False,
            indent=2,
        )

        self._state_file.write_text(
            serialized_content + "\n",
            encoding="utf-8",
        )

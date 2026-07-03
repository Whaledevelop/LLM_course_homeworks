import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from tutor_bot.application.recall_session_result import RecallSessionResult


class ActiveRecallHistoryRepository:
    def __init__(self, history_file: Path) -> None:
        self._history_file = history_file

    def save(
        self,
        result: RecallSessionResult,
        review_duration_seconds: float,
    ) -> None:
        record = {
            "attempt_id": str(uuid4()),
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "review_duration_seconds": round(review_duration_seconds, 3),
            "result": result.model_dump(mode="json"),
        }
        serialized_record = json.dumps(
            record,
            ensure_ascii=False,
        )

        self._history_file.parent.mkdir(parents=True, exist_ok=True)

        with self._history_file.open(
            "a",
            encoding="utf-8",
            newline="\n",
        ) as history_file:
            history_file.write(serialized_record + "\n")

    def load_results(self) -> tuple[RecallSessionResult, ...]:
        if not self._history_file.exists():
            return ()

        results = []

        for line in self._history_file.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            results.append(RecallSessionResult.model_validate(record["result"]))

        return tuple(results)

    def clear(self) -> None:
        self._history_file.unlink(missing_ok=True)

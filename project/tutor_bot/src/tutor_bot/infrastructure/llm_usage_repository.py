from pathlib import Path

from tutor_bot.generation.llm_response import LlmResponse


class LlmUsageRepository:
    def __init__(self, usage_file: Path) -> None:
        self._usage_file = usage_file

    def save(
        self,
        response: LlmResponse,
    ) -> None:
        self._usage_file.parent.mkdir(parents=True, exist_ok=True)

        with self._usage_file.open(
            "a",
            encoding="utf-8",
            newline="\n",
        ) as usage_file:
            usage_file.write(response.model_dump_json() + "\n")

    def load(self) -> tuple[LlmResponse, ...]:
        if not self._usage_file.exists():
            return ()

        return tuple(
            LlmResponse.model_validate_json(line)
            for line in self._usage_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

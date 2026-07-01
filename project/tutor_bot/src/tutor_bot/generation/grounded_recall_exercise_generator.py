from typing import Protocol

from tutor_bot.application.recall_exercise import RecallExercise


class GroundedRecallExerciseGenerator(Protocol):
    def generate(
        self,
        note_title: str,
        markdown_content: str,
    ) -> RecallExercise: ...

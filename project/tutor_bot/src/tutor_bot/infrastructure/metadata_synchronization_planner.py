from tutor_bot.application.metadata_synchronization_plan import (
    MetadataSynchronizationPlan,
)
from tutor_bot.infrastructure.note_consistency_checker import (
    NoteConsistencyChecker,
)


class MetadataSynchronizationPlanner:
    def __init__(
        self,
        consistency_checker: NoteConsistencyChecker,
    ) -> None:
        self._consistency_checker = consistency_checker

    def create_plan(self) -> MetadataSynchronizationPlan:
        report = self._consistency_checker.check()

        return MetadataSynchronizationPlan(
            metadata_note_ids_to_remove=(report.missing_markdown_note_ids),
            markdown_paths_to_register=(report.markdown_without_metadata),
            consistency_report=report,
        )

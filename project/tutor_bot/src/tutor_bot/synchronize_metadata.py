import argparse
import json
from collections.abc import Sequence

from tutor_bot.config import get_settings
from tutor_bot.infrastructure.metadata_synchronization_planner import (
    MetadataSynchronizationPlanner,
)
from tutor_bot.infrastructure.metadata_synchronization_service import (
    MetadataSynchronizationService,
)
from tutor_bot.infrastructure.note_consistency_checker import (
    NoteConsistencyChecker,
)
from tutor_bot.infrastructure.notes_metadata_repository import (
    NotesMetadataRepository,
)


def main(
    arguments: Sequence[str] | None = None,
) -> int:
    parsed_arguments = _parse_arguments(arguments)
    settings = get_settings()
    repository = NotesMetadataRepository(settings.metadata_file)
    checker = NoteConsistencyChecker(
        repository,
        settings.source_notes_dir,
    )
    planner = MetadataSynchronizationPlanner(checker)
    plan = planner.create_plan()

    _print_json(
        {
            "mode": "apply" if parsed_arguments.apply else "dry-run",
            "has_changes": plan.has_changes,
            "can_apply": plan.can_apply,
            "plan": plan.model_dump(mode="json"),
        }
    )

    if not plan.can_apply:
        return 2

    if not parsed_arguments.apply:
        if plan.has_changes:
            return 1

        return 0

    service = MetadataSynchronizationService(
        repository,
        planner,
        settings.source_notes_dir,
    )

    synchronized_catalog = service.synchronize()
    final_report = checker.check()

    _print_json(
        {
            "applied": plan.has_changes,
            "metadata_records": len(synchronized_catalog.notes),
            "is_consistent": final_report.is_consistent,
        }
    )

    if not final_report.is_consistent:
        return 2

    return 0


def _parse_arguments(
    arguments: Sequence[str] | None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check and synchronize Tutor Bot metadata.",
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the synchronization plan with metadata backup.",
    )

    return parser.parse_args(arguments)


def _print_json(
    value: dict[str, object],
) -> None:
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())

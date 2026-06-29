from tutor_bot.config import get_settings
from tutor_bot.infrastructure.note_consistency_checker import (
    NoteConsistencyChecker,
)
from tutor_bot.infrastructure.notes_metadata_repository import (
    NotesMetadataRepository,
)


def main() -> int:
    settings = get_settings()
    metadata_repository = NotesMetadataRepository(settings.metadata_file)

    checker = NoteConsistencyChecker(
        metadata_repository,
        settings.source_notes_dir,
    )

    report = checker.check()
    print(report.model_dump_json(indent=2))

    if not report.is_consistent:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

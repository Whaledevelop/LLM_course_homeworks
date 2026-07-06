from uuid import uuid4

from tutor_bot.infrastructure.markdown_note_storage import MarkdownNoteStorage


def test_create_uses_note_title_as_filename(tmp_path):
    storage = MarkdownNoteStorage(tmp_path)

    relative_path, _ = storage.create(
        uuid4(),
        "Память Unity: Managed Heap, Native Memory, Memory Profiler",
        "Содержание",
    )

    assert relative_path.as_posix() == (
        "_tutor_bot/Память Unity Managed Heap, Native Memory, Memory Profiler.md"
    )
    assert (tmp_path / relative_path).is_file()


def test_create_adds_suffix_when_filename_exists(tmp_path):
    storage = MarkdownNoteStorage(tmp_path)
    title = "Память Unity"

    first_path, _ = storage.create(uuid4(), title, "Первая заметка")
    second_path, _ = storage.create(uuid4(), title, "Вторая заметка")

    assert first_path.as_posix() == "_tutor_bot/Память Unity.md"
    assert second_path.as_posix() == "_tutor_bot/Память Unity (2).md"

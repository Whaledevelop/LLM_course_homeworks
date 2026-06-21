from pathlib import Path
import shutil

import chromadb

from settings import Settings


SUPPORTED_EXTENSIONS = {".md", ".txt"}


def save_uploaded_files(uploaded_files: list, settings: Settings) -> int:
    saved_count = 0
    for uploaded_file in uploaded_files:
        relative_path = Path(uploaded_file.name.replace("\\", "/"))
        if not _is_supported(relative_path):
            continue

        destination = settings.documents_directory / relative_path
        if ".." in relative_path.parts:
            destination = settings.documents_directory / relative_path.name

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(uploaded_file.getvalue())
        saved_count += 1

    return saved_count


def clear_documents(settings: Settings) -> None:
    chroma_client = chromadb.PersistentClient(path=str(settings.chroma_directory))
    for collection in chroma_client.list_collections():
        chroma_client.delete_collection(collection.name)

    if settings.documents_directory.exists():
        shutil.rmtree(settings.documents_directory)


def get_source_files(settings: Settings) -> list[Path]:
    source_files = []
    if settings.documents_directory.exists():
        source_files.extend(
            source_file
            for source_file in settings.documents_directory.rglob("*")
            if source_file.is_file() and _is_supported(source_file)
        )

    return source_files


def get_documents_context(settings: Settings) -> str:
    context_parts = []
    for source_file in get_source_files(settings):
        source_name = get_source_name(source_file, settings)
        context_parts.append(f"# {source_name}\n\n{source_file.read_text(encoding='utf-8')}")

    if not context_parts:
        context = "Документы пока не добавлены."
    else:
        context = "\n\n---\n\n".join(context_parts)

    return context


def get_source_name(source_file: Path, settings: Settings) -> str:
    source_name = source_file.relative_to(settings.documents_directory).as_posix()

    return source_name


def _is_supported(source_file: Path) -> bool:
    is_supported = source_file.suffix.lower() in SUPPORTED_EXTENSIONS

    return is_supported

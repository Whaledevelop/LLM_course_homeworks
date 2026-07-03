import os
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel


def write_json_atomically(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    content = model.model_dump_json(indent=2) + "\n"

    try:
        with temporary_file.open("w", encoding="utf-8", newline="\n") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())

        temporary_file.replace(path)
    finally:
        temporary_file.unlink(missing_ok=True)

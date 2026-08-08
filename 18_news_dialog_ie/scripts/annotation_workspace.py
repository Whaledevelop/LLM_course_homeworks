import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

from evaluation import annotation_template_fingerprint, load_gold_labels, write_annotation_template


def validate_gold(gold_path: Path, template_path: Path, progress_path: Path, gold_size: int, allow_incomplete: bool) -> None:
    template_dialog_ids = read_template_dialog_ids(template_path)
    if len(template_dialog_ids) != gold_size:
        raise ValueError(f"Annotation template contains {len(template_dialog_ids)} dialogs; exactly {gold_size} are required.")
    target_dialog_id_set = set(template_dialog_ids)
    gold_labels = load_gold_labels(gold_path)
    unexpected_dialog_ids = set(gold_labels) - target_dialog_id_set
    if unexpected_dialog_ids:
        raise ValueError(f"Gold data contains {len(unexpected_dialog_ids)} dialogs outside the current annotation template.")
    if allow_incomplete:
        return
    progress = read_annotation_progress(progress_path)
    expected_fingerprint = annotation_template_fingerprint(template_dialog_ids)
    if progress.get("template_fingerprint") != expected_fingerprint:
        raise ValueError("Annotation progress does not match the current template.")
    reviewed_dialog_ids = set(progress.get("reviewed_dialog_ids", [])) & target_dialog_id_set
    if len(reviewed_dialog_ids) < gold_size:
        raise ValueError(f"Only {len(reviewed_dialog_ids)} of {gold_size} annotation dialogs are reviewed.")


def prepare_annotation_workspace(data_dir: Path, dialog_ids: list[str]) -> None:
    template_path = data_dir / "gold_annotation_template.csv"
    gold_path = data_dir / "gold_annotations.csv"
    progress_path = data_dir / "annotation_progress.json"
    existing_dialog_ids = read_template_dialog_ids(template_path) if template_path.exists() else []
    if existing_dialog_ids != dialog_ids:
        backup_annotation_files(data_dir, (template_path, gold_path, progress_path))
        write_annotation_template(template_path, dialog_ids)
        write_empty_gold(gold_path)
        write_annotation_progress(progress_path, dialog_ids, [])
        return
    progress = read_annotation_progress(progress_path)
    reviewed_dialog_ids = [dialog_id for dialog_id in progress.get("reviewed_dialog_ids", []) if dialog_id in set(dialog_ids)]
    write_annotation_progress(progress_path, dialog_ids, reviewed_dialog_ids)


def read_template_dialog_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        return [row["dialog_id"].strip() for row in reader if row.get("dialog_id", "").strip()]


def read_annotation_progress(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:

        return json.load(file)


def write_annotation_progress(path: Path, dialog_ids: list[str], reviewed_dialog_ids: list[str]) -> None:
    reviewed = [dialog_id for dialog_id in dialog_ids if dialog_id in set(reviewed_dialog_ids)]
    payload = {"template_fingerprint": annotation_template_fingerprint(dialog_ids), "reviewed_dialog_ids": reviewed}
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_empty_gold(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["dialog_id", "label", "value"])
        writer.writeheader()


def backup_annotation_files(data_dir: Path, paths: tuple[Path, ...]) -> None:
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        return
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup_dir = data_dir / "annotation_backups" / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in existing_paths:
        shutil.copy2(path, backup_dir / path.name)

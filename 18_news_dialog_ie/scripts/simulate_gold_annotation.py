from pathlib import Path

from annotation_app import load_dialogs, load_template_ids, save_annotations, save_reviewed
from annotation_workspace import backup_annotation_files
from evaluation import annotation_template_fingerprint


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
DIALOGS_PATH = DATA_DIR / "news_dialogs.jsonl"
TEMPLATE_PATH = DATA_DIR / "gold_annotation_template.csv"
GOLD_PATH = DATA_DIR / "gold_annotations.csv"
PROGRESS_PATH = DATA_DIR / "annotation_progress.json"
EXPECTED_DIALOG_IDS = [
    "f191c458060d6a71b149e9e208da99b9",
    "ca60ed06c5383a34e2aca78f92d70bf7",
    "ec9c53d87f1707cfece8c482091877bd",
    "9772d6bfe6c04c1c56b943134b12b236",
    "1c32602338388652ec73f7e713f34965",
    "82a843f7b85e24575e3edca138384a34",
    "1f96fa2008341fcf0c31ab9ca893e1f8",
    "1d88fe9cf6440ba79afada9f0036804a",
    "d7c98db3a2008d818812b2ad9a8acd43",
    "865b21a4b38b6c3885b5a0dda054163e",
]
ANNOTATION_SPECS = {
    "f191c458060d6a71b149e9e208da99b9": [
        ("ORG", "Amazon"),
        ("ORG", "Amazon Web Services"),
        ("ORG", "AWS"),
        ("ORG", "Alexa"),
        ("ORG", "Prime Air"),
        ("ORG", "Amazon Go"),
        ("ORG", "Amazon Prime Video"),
        ("ORG", "Amazon Fresh"),
        ("ORG", "Whole Foods Market"),
        ("LOC", "India"),
        ("LOC", "Europe"),
        ("LOC", "Latin America"),
        ("DATE", "past 5 years"),
        ("DATE", "2020"),
        ("DATE", "2017"),
        ("DATE", "2018"),
        ("DATE", "2019"),
        ("DATE", "2021"),
        ("SOURCE", "Forbes"),
        ("SOURCE", "Business Insider"),
        ("SOURCE", "CNBC"),
        ("SOURCE", "The Wall Street Journal"),
        ("SOURCE", "Reuters"),
        ("SOURCE", "The New York Times"),
    ],
    "ca60ed06c5383a34e2aca78f92d70bf7": [
        ("LOC", "India"),
    ],
    "ec9c53d87f1707cfece8c482091877bd": [
        ("PERSON", "Jonathan Riley-Smith"),
        ("PERSON", "King Louis IX"),
        ("ORG", "Ignatius Press"),
        ("ORG", "Catholic Church"),
        ("LOC", "Middle East"),
        ("LOC", "Jerusalem"),
        ("LOC", "Europe"),
        ("DATE", "March 23, 2009"),
        ("EVENT", "First through Fifth Crusades"),
        ("EVENT", "French crusades"),
    ],
    "9772d6bfe6c04c1c56b943134b12b236": [
        ("LOC", "Allen, Texas"),
        ("LOC", "Dallas"),
        ("EVENT", "shooting at an outlet mall"),
        ("IMPACT", "at least one confirmed shooter who is being reported as deceased"),
        ("SOURCE", "CNN"),
    ],
    "1c32602338388652ec73f7e713f34965": [
        ("PERSON", "Jonathan Riley-Smith"),
        ("PERSON", "King Louis IX"),
        ("PERSON", "Pope Urban II"),
        ("ORG", "Ignatius Press"),
        ("ORG", "Catholic Church"),
        ("LOC", "Middle East"),
        ("LOC", "Jerusalem"),
        ("LOC", "Europe"),
        ("LOC", "Holy Land"),
        ("DATE", "March 23, 2009"),
        ("DATE", "1095"),
        ("EVENT", "First through Fifth Crusades"),
        ("EVENT", "French crusades"),
    ],
    "82a843f7b85e24575e3edca138384a34": [
        ("LOC", "South Eastern Nebraska"),
        ("LOC", "Topeka KS"),
        ("LOC", "Morris County"),
        ("LOC", "Kansas"),
        ("LOC", "Wabaunsee County"),
        ("LOC", "Lyon County"),
        ("LOC", "Omaha/Valley NE"),
        ("LOC", "Seward County"),
        ("LOC", "Nebraska"),
        ("LOC", "Saline County"),
        ("LOC", "Butler County"),
        ("LOC", "Lushton"),
        ("LOC", "Geneva"),
        ("LOC", "Ulysses"),
        ("LOC", "Beaver Crossing"),
        ("DATE", "933 PM CDT Thu Aug 15 2019"),
        ("DATE", "10:00 PM CDT"),
        ("DATE", "1128 PM CDT Sat May 15 2021"),
        ("DATE", "midnight CDT"),
        ("EVENT", "Tornado Warning"),
        ("EVENT", "Civil Danger Warning"),
        ("IMPACT", "Flying debris will be dangerous"),
        ("IMPACT", "Mobile homes will be damaged or destroyed"),
        ("IMPACT", "Damage to roofs, windows, and vehicles will occur"),
        ("IMPACT", "damage is likely"),
        ("IMPACT", "No structure above ground will be able to survive this storm"),
        ("SOURCE", "National Weather Service Topeka KS"),
        ("SOURCE", "National Weather Service Omaha/Valley NE"),
        ("SOURCE", "Weather spotters"),
    ],
    "1f96fa2008341fcf0c31ab9ca893e1f8": [
        ("PERSON", "Joe Biden"),
        ("DATE", "Easter of 2023"),
        ("EVENT", "Easter egg rolls"),
    ],
    "1d88fe9cf6440ba79afada9f0036804a": [
        ("PERSON", "Matthew Kacsmaryk"),
        ("PERSON", "Trump"),
        ("PERSON", "Thomas O. Rice"),
        ("PERSON", "Obama"),
        ("PERSON", "President Biden"),
        ("ORG", "Food and Drug Administration"),
        ("ORG", "US Supreme Court"),
        ("ORG", "FDA"),
        ("LOC", "US"),
        ("LOC", "Texas"),
        ("LOC", "Washington"),
        ("DATE", "Friday"),
        ("DATE", "2000"),
        ("DATE", "more than 20 years"),
        ("DATE", "almost a year"),
        ("DATE", "seven days"),
        ("DATE", "past 23 years"),
        ("EVENT", "conflicting court rulings"),
        ("EVENT", "overturned Roe v. Wade"),
        ("IMPACT", "Access to the most commonly used method of abortion in the US plunged into uncertainty"),
        ("IMPACT", "restrict access to the drug in at least 17 states"),
        ("SOURCE", "AP"),
    ],
    "d7c98db3a2008d818812b2ad9a8acd43": [
        ("PERSON", "Mike Lindell"),
        ("PERSON", "Robert Zeidman"),
        ("PERSON", "Brian Glasser"),
        ("ORG", "MyPillow"),
        ("ORG", "Lindell Management"),
        ("LOC", "South Dakota"),
        ("LOC", "Nevada"),
        ("DATE", "August 2021"),
        ("DATE", "Wednesday"),
        ("DATE", "2020 election"),
        ("DATE", "November 2020"),
        ("DATE", "30 days"),
        ("EVENT", "private arbitration panel ruled"),
        ("IMPACT", "$5 million payout"),
        ("SOURCE", "The Washington Post"),
    ],
    "865b21a4b38b6c3885b5a0dda054163e": [],
}


def build_annotations(dialog_ids: list[str], dialogs: dict[str, str]) -> list[dict[str, str]]:
    annotations = []
    for dialog_id in dialog_ids:
        text = dialogs[dialog_id]
        seen = set()
        for label, value in ANNOTATION_SPECS[dialog_id]:
            start = text.find(value)
            if start < 0:
                raise ValueError(f"Span not found for {dialog_id}: {label}={value!r}")
            end = start + len(value)
            if text[start:end] != value:
                raise ValueError(f"Invalid offsets for {dialog_id}: {start}:{end}")
            key = (dialog_id, label, value)
            if key in seen:
                continue
            seen.add(key)
            annotations.append({"dialog_id": dialog_id, "label": label, "value": value})

    return annotations


def simulate_annotation(data_dir: Path = DATA_DIR, create_backup: bool = True) -> tuple[int, int]:
    dialogs_path = data_dir / DIALOGS_PATH.name
    template_path = data_dir / TEMPLATE_PATH.name
    gold_path = data_dir / GOLD_PATH.name
    progress_path = data_dir / PROGRESS_PATH.name
    dialog_ids = load_template_ids(template_path)
    if dialog_ids != EXPECTED_DIALOG_IDS:
        raise ValueError("The current annotation template does not match the expected 10 gold dialogs.")
    dialogs = load_dialogs(dialogs_path)
    missing_dialog_ids = [dialog_id for dialog_id in dialog_ids if dialog_id not in dialogs]
    if missing_dialog_ids:
        raise ValueError(f"Dialogs missing from JSONL: {', '.join(missing_dialog_ids)}")
    annotations = build_annotations(dialog_ids, dialogs)
    if create_backup:
        backup_annotation_files(data_dir, (gold_path, progress_path))
    reviewed_dialog_ids = set()
    template_fingerprint = annotation_template_fingerprint(dialog_ids)
    for dialog_id in dialog_ids:
        saved_annotations = [row for row in annotations if row["dialog_id"] in reviewed_dialog_ids | {dialog_id}]
        save_annotations(gold_path, saved_annotations)
        reviewed_dialog_ids.add(dialog_id)
        save_reviewed(progress_path, reviewed_dialog_ids, dialog_ids, template_fingerprint)

    return len(annotations), len(reviewed_dialog_ids)


def main() -> None:
    annotation_count, reviewed_count = simulate_annotation()
    print(f"Gold annotations saved: {annotation_count}")
    print(f"Dialogs reviewed: {reviewed_count}/{len(EXPECTED_DIALOG_IDS)}")
    print(f"Gold file: {GOLD_PATH}")
    print(f"Progress file: {PROGRESS_PATH}")


if __name__ == "__main__":
    main()

from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from tutor_bot.application.vacancy import Vacancy
from tutor_bot.application.vacancy_analysis import VacancyAnalysis
from tutor_bot.infrastructure.atomic_json import write_json_atomically


class VacancyRepository:
    def __init__(self, vacancies_dir: Path, db_id: str) -> None:
        self._vacancies_dir = vacancies_dir / db_id

    def save_pdf(
        self,
        filename: str,
        pdf_content: bytes,
        analysis: VacancyAnalysis,
    ) -> Vacancy:
        content_hash = sha256(pdf_content).hexdigest()
        existing_vacancy = self.find_by_sha256(content_hash)
        vacancy_id = existing_vacancy.id if existing_vacancy is not None else uuid4()
        vacancy = Vacancy(
            id=vacancy_id,
            sha256=content_hash,
            original_filename=Path(filename).name,
            title=analysis.title,
            uploaded_at=datetime.now(timezone.utc),
            extracted_text=self.extract_text(pdf_content),
            requirements=analysis.requirements,
        )
        vacancy_dir = self._vacancies_dir / str(vacancy_id)
        vacancy_dir.mkdir(parents=True, exist_ok=True)
        self._write_pdf(vacancy_dir / "vacancy.pdf", pdf_content)
        write_json_atomically(vacancy_dir / "vacancy.json", vacancy)

        return vacancy

    def list_vacancies(self) -> tuple[Vacancy, ...]:
        if not self._vacancies_dir.exists():
            return ()

        vacancies = []

        for vacancy_file in self._vacancies_dir.glob("*/vacancy.json"):
            vacancies.append(Vacancy.model_validate_json(vacancy_file.read_text(encoding="utf-8-sig")))

        return tuple(sorted(vacancies, key=lambda vacancy: vacancy.uploaded_at, reverse=True))

    def get(self, vacancy_id: UUID) -> Vacancy:
        vacancy_file = self._vacancies_dir / str(vacancy_id) / "vacancy.json"

        if not vacancy_file.exists():
            raise ValueError(f"Vacancy {vacancy_id} was not found")

        return Vacancy.model_validate_json(vacancy_file.read_text(encoding="utf-8-sig"))

    def find_by_sha256(self, content_hash: str) -> Vacancy | None:
        return next(
            (vacancy for vacancy in self.list_vacancies() if vacancy.sha256 == content_hash),
            None,
        )

    @staticmethod
    def extract_text(pdf_content: bytes) -> str:
        try:
            reader = PdfReader(BytesIO(pdf_content))
            text = "\n\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()
        except (PdfReadError, OSError, ValueError) as error:
            raise ValueError("Не удалось прочитать PDF-файл") from error

        if not text:
            raise ValueError("PDF не содержит текстового слоя. OCR не поддерживается")

        return text

    @staticmethod
    def _write_pdf(path: Path, content: bytes) -> None:
        temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")

        try:
            temporary_path.write_bytes(content)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)

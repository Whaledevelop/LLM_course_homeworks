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
        return self.save_document(filename, pdf_content, analysis)

    def save_document(
        self,
        filename: str,
        content: bytes,
        analysis: VacancyAnalysis,
    ) -> Vacancy:
        source_format = self._get_source_format(filename)
        content_hash = sha256(content).hexdigest()
        existing_vacancy = self.find_by_sha256(content_hash)
        vacancy_id = existing_vacancy.id if existing_vacancy is not None else uuid4()
        vacancy = Vacancy(
            id=vacancy_id,
            sha256=content_hash,
            original_filename=Path(filename).name,
            source_format=source_format,
            title=analysis.title,
            uploaded_at=datetime.now(timezone.utc),
            extracted_text=self.extract_text(filename, content),
            requirements=analysis.requirements,
        )
        vacancy_dir = self._vacancies_dir / str(vacancy_id)
        vacancy_dir.mkdir(parents=True, exist_ok=True)
        self._write_source(vacancy_dir / f"vacancy.{source_format}", content)
        stale_format = "md" if source_format == "pdf" else "pdf"
        (vacancy_dir / f"vacancy.{stale_format}").unlink(missing_ok=True)
        write_json_atomically(vacancy_dir / "vacancy.json", vacancy)

        return vacancy

    def list_vacancies(self) -> tuple[Vacancy, ...]:
        if not self._vacancies_dir.exists():
            return ()

        vacancies = []

        for vacancy_file in self._vacancies_dir.glob("*/vacancy.json"):
            vacancies.append(
                Vacancy.model_validate_json(vacancy_file.read_text(encoding="utf-8-sig"))
            )

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
    def extract_text(filename: str, content: bytes) -> str:
        source_format = VacancyRepository._get_source_format(filename)

        if source_format == "md":
            return VacancyRepository._extract_markdown_text(content)

        return VacancyRepository.extract_pdf_text(content)

    @staticmethod
    def extract_pdf_text(pdf_content: bytes) -> str:
        try:
            reader = PdfReader(BytesIO(pdf_content))
            text = "\n\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()
        except (PdfReadError, OSError, ValueError) as error:
            raise ValueError("Не удалось прочитать PDF-файл") from error

        if not text:
            raise ValueError("PDF не содержит текстового слоя. OCR не поддерживается")

        return text

    @staticmethod
    def _extract_markdown_text(content: bytes) -> str:
        try:
            text = content.decode("utf-8-sig").strip()
        except UnicodeDecodeError as error:
            raise ValueError("MD-файл должен быть сохранен в UTF-8") from error

        if not text:
            raise ValueError("MD-файл не содержит текста")

        return text

    @staticmethod
    def _get_source_format(filename: str) -> str:
        source_format = Path(filename).suffix.lower().lstrip(".")

        if source_format not in {"pdf", "md"}:
            raise ValueError("Поддерживаются только PDF и MD файлы")

        return source_format

    @staticmethod
    def _write_source(path: Path, content: bytes) -> None:
        temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")

        try:
            temporary_path.write_bytes(content)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)

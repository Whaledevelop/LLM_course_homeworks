from pathlib import Path
import shutil
import tempfile
from typing import BinaryIO

from pdf_service import PdfConversionResult
from pdf_service import PdfService
from settings import Settings


class DocumentService:
    def __init__(self) -> None:
        self._pdf_service = PdfService()

    def save_and_convert_uploaded_pdfs(
        self,
        uploaded_files: list[BinaryIO],
    ) -> list[PdfConversionResult]:
        Settings.ensure_directories()

        conversion_results: list[PdfConversionResult] = []

        for uploaded_file in uploaded_files:
            uploaded_file_name = Path(uploaded_file.name).name
            source_pdf_path: Path | None = None

            with tempfile.NamedTemporaryFile(
                suffix=".pdf",
                delete=False,
            ) as temporary_file:
                temporary_file.write(uploaded_file.getbuffer())
                temporary_pdf_path = Path(temporary_file.name)

            try:
                source_pdf_path = temporary_pdf_path.with_name(uploaded_file_name)
                temporary_pdf_path.rename(source_pdf_path)

                conversion_result = self._pdf_service.convert_pdf_to_markdown(
                    source_pdf_path=source_pdf_path,
                    documents_dir=Settings.DOCUMENTS_DIR,
                )

                conversion_results.append(conversion_result)
            finally:
                if source_pdf_path and source_pdf_path.exists():
                    source_pdf_path.unlink()

        return conversion_results

    def get_uploaded_pdf_paths(self) -> list[Path]:
        Settings.ensure_directories()

        pdf_paths = sorted(Settings.DOCUMENTS_DIR.glob("*.pdf"))

        return pdf_paths

    def get_markdown_paths(self) -> list[Path]:
        Settings.ensure_directories()

        markdown_paths = sorted(Settings.DOCUMENTS_DIR.glob("*.md"))

        return markdown_paths

    def read_markdown(self, markdown_path: Path) -> str:
        markdown_text = markdown_path.read_text(
            encoding="utf-8",
        )

        return markdown_text

    def clear_documents_and_index(self) -> None:
        if Settings.DOCUMENTS_DIR.exists():
            shutil.rmtree(Settings.DOCUMENTS_DIR)

        if Settings.CHROMA_DIR.exists():
            shutil.rmtree(Settings.CHROMA_DIR)

        Settings.ensure_directories()

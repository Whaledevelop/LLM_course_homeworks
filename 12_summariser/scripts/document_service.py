from pathlib import Path
import re
import tempfile
from typing import BinaryIO

from models import ConversionResult
from pdf_service import PdfService
from content_filter_service import ContentFilterService


class DocumentService:
    def __init__(self) -> None:
        self._pdf_service = PdfService()
        self._content_filter_service = ContentFilterService()

    def convert_uploaded_pdfs(
        self,
        uploaded_files: list[BinaryIO],
        output_dir: Path,
        include_metadata: bool,
        include_images: bool,
    ) -> list[ConversionResult]:
        results: list[ConversionResult] = []
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for uploaded_file in uploaded_files:
            result = self._convert_uploaded_pdf(
                uploaded_file=uploaded_file,
                output_dir=output_dir,
                include_metadata=include_metadata,
                include_images=include_images,
            )
            results.append(result)

        return results

    def _convert_uploaded_pdf(
        self,
        uploaded_file: BinaryIO,
        output_dir: Path,
        include_metadata: bool,
        include_images: bool,
    ) -> ConversionResult:
        source_pdf_path = self._create_temporary_pdf(uploaded_file)

        try:
            pdf_path, pages, images_count = self._pdf_service.extract_content(
                source_pdf_path=source_pdf_path,
                output_dir=output_dir,
                include_images=include_images,
            )
            markdown = self._content_filter_service.create_markdown(
                document_title=pdf_path.stem,
                pages=pages,
                include_metadata=include_metadata,
            )
            markdown = self._remove_page_headers(markdown)
            markdown_path = output_dir / f"{pdf_path.stem}.md"
            markdown_path.write_text(
                markdown,
                encoding="utf-8",
            )
        finally:
            source_pdf_path.unlink(
                missing_ok=True,
            )

        result = ConversionResult(
            pdf_path=pdf_path,
            markdown_path=markdown_path,
            images_count=images_count,
            pages_count=len(pages),
        )

        return result

    def _remove_page_headers(
        self,
        markdown: str,
    ) -> str:
        return re.sub(
            r"(?m)^## (?:Страница|Page)\s+\d+\s*\n+",
            "",
            markdown,
        )

    def _create_temporary_pdf(
        self,
        uploaded_file: BinaryIO,
    ) -> Path:
        uploaded_file_name = Path(uploaded_file.name).name

        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as temporary_file:
            temporary_file.write(uploaded_file.getbuffer())
            temporary_pdf_path = Path(temporary_file.name)

        source_pdf_path = temporary_pdf_path.with_name(uploaded_file_name)
        temporary_pdf_path.rename(source_pdf_path)

        return source_pdf_path

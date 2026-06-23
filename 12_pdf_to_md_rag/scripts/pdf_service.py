from dataclasses import dataclass
from pathlib import Path
import hashlib
import shutil

import fitz


@dataclass
class PdfConversionResult:
    pdf_path: Path
    markdown_path: Path
    pages_count: int
    images_count: int


@dataclass
class PageElement:
    y_position: float
    markdown: str


class PdfService:
    browser_supported_extensions = {
        "png",
        "jpg",
        "jpeg",
        "gif",
        "webp",
    }

    def convert_pdf_to_markdown(
        self,
        source_pdf_path: Path,
        documents_dir: Path,
    ) -> PdfConversionResult:
        document_name = source_pdf_path.stem
        target_pdf_path = documents_dir / source_pdf_path.name
        markdown_path = documents_dir / f"{document_name}.md"
        document_assets_dir = documents_dir / f"{document_name}_assets"
        images_dir = document_assets_dir / "images"

        self._prepare_output_paths(
            source_pdf_path=source_pdf_path,
            target_pdf_path=target_pdf_path,
            markdown_path=markdown_path,
            document_assets_dir=document_assets_dir,
        )

        pdf_document = fitz.open(target_pdf_path)
        pages_count = pdf_document.page_count
        markdown_parts: list[str] = [f"# {document_name}", ""]
        extracted_image_hashes: set[str] = set()
        total_images_count = 0

        for page_index in range(pages_count):
            page = pdf_document.load_page(page_index)
            page_number = page_index + 1

            markdown_parts.append(f"## Страница {page_number}")
            markdown_parts.append("")

            page_elements = self._extract_page_elements(
                pdf_document=pdf_document,
                page=page,
                page_number=page_number,
                document_name=document_name,
                images_dir=images_dir,
                extracted_image_hashes=extracted_image_hashes,
            )

            image_elements_count = len(
                [
                    element
                    for element in page_elements
                    if element.markdown.startswith("![")
                ]
            )

            total_images_count += image_elements_count

            if not page_elements:
                markdown_parts.append(
                    "> На странице не обнаружен текстовый слой. Для распознавания требуется OCR."
                )
                markdown_parts.append("")
                continue

            for element in page_elements:
                markdown_parts.append(element.markdown)
                markdown_parts.append("")

        pdf_document.close()

        markdown_path.write_text(
            "\n".join(markdown_parts),
            encoding="utf-8",
        )

        return PdfConversionResult(
            pdf_path=target_pdf_path,
            markdown_path=markdown_path,
            pages_count=pages_count,
            images_count=total_images_count,
        )

    def _prepare_output_paths(
        self,
        source_pdf_path: Path,
        target_pdf_path: Path,
        markdown_path: Path,
        document_assets_dir: Path,
    ) -> None:
        target_pdf_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if markdown_path.exists():
            markdown_path.unlink()

        if document_assets_dir.exists():
            shutil.rmtree(document_assets_dir)

        if source_pdf_path.resolve() != target_pdf_path.resolve():
            shutil.copy2(
                source_pdf_path,
                target_pdf_path,
            )

    def _extract_page_elements(
        self,
        pdf_document: fitz.Document,
        page: fitz.Page,
        page_number: int,
        document_name: str,
        images_dir: Path,
        extracted_image_hashes: set[str],
    ) -> list[PageElement]:
        page_elements: list[PageElement] = []

        text_blocks = page.get_text("blocks")

        for block in text_blocks:
            block_text = str(block[4]).strip()

            if not block_text:
                continue

            page_elements.append(
                PageElement(
                    y_position=float(block[1]),
                    markdown=block_text,
                )
            )

        image_infos = page.get_images(full=True)
        image_number_on_page = 0

        for image_info in image_infos:
            xref = image_info[0]
            image_rects = page.get_image_rects(xref)

            if not image_rects:
                continue

            image_bytes = pdf_document.extract_image(xref)["image"]
            image_hash = hashlib.sha256(image_bytes).hexdigest()

            if image_hash in extracted_image_hashes:
                continue

            extracted_image_hashes.add(image_hash)
            image_number_on_page += 1

            image_extension = self._get_image_extension(
                pdf_document=pdf_document,
                xref=xref,
            )

            images_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            image_file_name = (
                f"page-{page_number:03d}-image-{image_number_on_page:03d}.{image_extension}"
            )

            image_path = images_dir / image_file_name

            self._save_image(
                pdf_document=pdf_document,
                xref=xref,
                image_path=image_path,
                image_extension=image_extension,
            )

            relative_image_path = (
                f"{document_name}_assets/images/{image_file_name}"
            )

            page_elements.append(
                PageElement(
                    y_position=float(image_rects[0].y0),
                    markdown=(
                        f"![Изображение {image_number_on_page} со страницы {page_number}]"
                        f"({relative_image_path})"
                    ),
                )
            )

        page_elements.sort(
            key=lambda element: element.y_position,
        )

        return page_elements

    def _get_image_extension(
        self,
        pdf_document: fitz.Document,
        xref: int,
    ) -> str:
        image_info = pdf_document.extract_image(xref)
        image_extension = image_info.get("ext", "png").lower()

        if image_extension in self.browser_supported_extensions:

            return image_extension

        return "png"

    def _save_image(
        self,
        pdf_document: fitz.Document,
        xref: int,
        image_path: Path,
        image_extension: str,
    ) -> None:
        image_info = pdf_document.extract_image(xref)

        if image_extension == image_info.get("ext", "").lower():
            image_path.write_bytes(image_info["image"])

            return

        pixmap = fitz.Pixmap(pdf_document, xref)

        if pixmap.n >= 5:
            pixmap = fitz.Pixmap(fitz.csRGB, pixmap)

        pixmap.save(image_path)

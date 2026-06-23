import hashlib
from io import BytesIO
import shutil
from pathlib import Path

import fitz
import pytesseract
from PIL import Image
from PIL import ImageOps
from pytesseract import Output

from models import PageContent


class PdfService:
    _browser_supported_extensions = {
        "png",
        "jpg",
        "jpeg",
        "gif",
        "webp",
    }

    def extract_content(
        self,
        source_pdf_path: Path,
        output_dir: Path,
        include_images: bool,
    ) -> tuple[Path, list[PageContent], int]:
        document_name = source_pdf_path.stem
        target_pdf_path = output_dir / source_pdf_path.name
        assets_dir = output_dir / f"{document_name}_assets"
        images_dir = assets_dir / "images"

        self._prepare_output_paths(
            source_pdf_path=source_pdf_path,
            target_pdf_path=target_pdf_path,
            assets_dir=assets_dir,
        )

        pdf_document = fitz.open(target_pdf_path)
        extracted_hashes: set[str] = set()
        pages: list[PageContent] = []
        images_count = 0

        try:
            for page_index in range(pdf_document.page_count):
                page = pdf_document.load_page(page_index)
                page_content, page_images_count = self._extract_page_content(
                    pdf_document=pdf_document,
                    page=page,
                    page_number=page_index + 1,
                    document_name=document_name,
                    images_dir=images_dir,
                    extracted_hashes=extracted_hashes,
                    include_images=include_images,
                )
                pages.append(page_content)
                images_count += page_images_count
        finally:
            pdf_document.close()

        return target_pdf_path, pages, images_count

    def _prepare_output_paths(
        self,
        source_pdf_path: Path,
        target_pdf_path: Path,
        assets_dir: Path,
    ) -> None:
        target_pdf_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if assets_dir.exists():
            shutil.rmtree(assets_dir)

        if source_pdf_path.resolve() != target_pdf_path.resolve():
            shutil.copy2(source_pdf_path, target_pdf_path)

    def _extract_page_content(
        self,
        pdf_document: fitz.Document,
        page: fitz.Page,
        page_number: int,
        document_name: str,
        images_dir: Path,
        extracted_hashes: set[str],
        include_images: bool,
    ) -> tuple[PageContent, int]:
        text = page.get_text("text").strip()
        image_markdown: list[str] = []
        images_count = 0

        if not include_images:
            page_content = PageContent(
                number=page_number,
                text=text,
                image_markdown=image_markdown,
            )

            return page_content, images_count

        for image_number, image_info in enumerate(page.get_images(full=True), start=1):
            xref = image_info[0]
            image_bytes = pdf_document.extract_image(xref)["image"]
            image_hash = hashlib.sha256(image_bytes).hexdigest()

            if image_hash in extracted_hashes:
                continue

            extracted_hashes.add(image_hash)
            image_text = self._extract_image_text(image_bytes)

            if image_text:
                text = self._append_text(
                    text=text,
                    addition=image_text,
                )
                continue

            image_extension = self._get_image_extension(
                pdf_document=pdf_document,
                xref=xref,
            )
            image_path = images_dir / f"page-{page_number:03d}-image-{image_number:03d}.{image_extension}"
            images_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
            self._save_image(
                pdf_document=pdf_document,
                xref=xref,
                image_path=image_path,
                image_extension=image_extension,
            )

            relative_image_path = f"{document_name}_assets/images/{image_path.name}"
            image_markdown.append(
                f"![Изображение {image_number} со страницы {page_number}]({relative_image_path})"
            )
            images_count += 1

        page_content = PageContent(
            number=page_number,
            text=text,
            image_markdown=image_markdown,
        )

        return page_content, images_count

    def _extract_image_text(
        self,
        image_bytes: bytes,
    ) -> str:
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                grayscale_image = ImageOps.grayscale(image)
                ocr_data = pytesseract.image_to_data(
                    grayscale_image,
                    lang="rus+eng",
                    config="--psm 6",
                    output_type=Output.DICT,
                )
        except (OSError, pytesseract.TesseractNotFoundError):
            return ""

        words = [
            text.strip()
            for text, confidence in zip(ocr_data["text"], ocr_data["conf"])
            if text.strip() and float(confidence) >= 50
        ]
        recognized_text = " ".join(words)

        if len(words) < 5 or len(recognized_text) < 30:
            return ""

        return recognized_text

    def _append_text(
        self,
        text: str,
        addition: str,
    ) -> str:
        if not text:
            return addition

        return f"{text}\n\n{addition}"

    def _get_image_extension(
        self,
        pdf_document: fitz.Document,
        xref: int,
    ) -> str:
        image_extension = pdf_document.extract_image(xref).get("ext", "png").lower()

        if image_extension in self._browser_supported_extensions:
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

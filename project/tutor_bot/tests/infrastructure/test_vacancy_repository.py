from io import BytesIO

import pytest
from pypdf import PdfWriter

from tutor_bot.application.vacancy_analysis import VacancyAnalysis
from tutor_bot.application.vacancy_requirement import VacancyRequirement
from tutor_bot.infrastructure.vacancy_repository import VacancyRepository


def test_save_updates_duplicate_and_isolates_databases(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        VacancyRepository,
        "extract_text",
        staticmethod(lambda filename, content: "Unity ECS vacancy"),
    )
    analysis = _create_analysis()
    repository = VacancyRepository(tmp_path, "unity")
    first = repository.save_pdf("first.pdf", b"same-pdf", analysis)
    second = repository.save_pdf("renamed.pdf", b"same-pdf", analysis)

    assert first.id == second.id
    assert second.original_filename == "renamed.pdf"
    assert second.source_format == "pdf"
    assert len(repository.list_vacancies()) == 1
    assert VacancyRepository(tmp_path, "other").list_vacancies() == ()


def test_save_markdown_creates_json_and_source_file(tmp_path) -> None:
    repository = VacancyRepository(tmp_path, "unity")
    vacancy = repository.save_document(
        "unity.md",
        "# Unity Developer\n\nKnowledge of ECS".encode(),
        _create_analysis(),
    )
    vacancy_dir = tmp_path / "unity" / str(vacancy.id)

    assert vacancy.source_format == "md"
    assert vacancy.extracted_text.startswith("# Unity Developer")
    assert (vacancy_dir / "vacancy.md").exists()
    assert (vacancy_dir / "vacancy.json").exists()


def test_extract_text_rejects_corrupted_pdf() -> None:
    with pytest.raises(ValueError, match="прочитать"):
        VacancyRepository.extract_text("vacancy.pdf", b"not-a-pdf")


def test_extract_text_rejects_pdf_without_text_layer() -> None:
    output = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(output)

    with pytest.raises(ValueError, match="текстового слоя"):
        VacancyRepository.extract_text("vacancy.pdf", output.getvalue())


def test_extract_text_rejects_empty_markdown() -> None:
    with pytest.raises(ValueError, match="не содержит текста"):
        VacancyRepository.extract_text("vacancy.md", b"\xef\xbb\xbf\n")


def _create_analysis() -> VacancyAnalysis:
    return VacancyAnalysis(
        title="Unity Developer",
        requirements=(
            VacancyRequirement(
                topic="Unity ECS",
                expected_knowledge="Understand entities and systems",
                evidence="Knowledge of Unity ECS",
            ),
        ),
    )

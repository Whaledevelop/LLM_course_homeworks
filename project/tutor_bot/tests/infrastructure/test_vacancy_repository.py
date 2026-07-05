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
        staticmethod(lambda content: "Unity ECS vacancy"),
    )
    analysis = VacancyAnalysis(
        title="Unity Developer",
        requirements=(
            VacancyRequirement(
                topic="Unity ECS",
                expected_knowledge="Understand entities and systems",
                evidence="Knowledge of Unity ECS",
            ),
        ),
    )
    repository = VacancyRepository(tmp_path, "unity")
    first = repository.save_pdf("first.pdf", b"same-pdf", analysis)
    second = repository.save_pdf("renamed.pdf", b"same-pdf", analysis)

    assert first.id == second.id
    assert second.original_filename == "renamed.pdf"
    assert len(repository.list_vacancies()) == 1
    assert VacancyRepository(tmp_path, "other").list_vacancies() == ()


def test_extract_text_rejects_corrupted_pdf() -> None:
    with pytest.raises(ValueError, match="прочитать"):
        VacancyRepository.extract_text(b"not-a-pdf")


def test_extract_text_rejects_pdf_without_text_layer() -> None:
    output = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(output)

    with pytest.raises(ValueError, match="текстового слоя"):
        VacancyRepository.extract_text(output.getvalue())

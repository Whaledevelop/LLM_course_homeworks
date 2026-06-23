import re

from langchain_ollama import ChatOllama

from models import PageContent
from settings import Settings


class ContentFilterService:
    _metadata_markers = (
        "преподаватель",
        "лектор",
        "спикер",
        "ведущий вебинара",
        "карта курса",
        "программа курса",
        "план курса",
        "содержание курса",
        "маршрут вебинара",
        "цели вебинара",
        "цели занятия",
        "домашнее задание",
        "практическое задание",
        "контакты",
        "github.com",
        "t.me/",
        "telegram",
        "телеграм",
        "спасибо за внимание",
        "вопросы?",
    )

    def __init__(self) -> None:
        self._chat_model = ChatOllama(
            model=Settings.OLLAMA_CHAT_MODEL,
            base_url=Settings.OLLAMA_BASE_URL,
            temperature=0,
        )

    def create_markdown(
        self,
        document_title: str,
        pages: list[PageContent],
        include_metadata: bool,
    ) -> str:
        markdown_parts = [f"# {document_title}", ""]

        for page in pages:
            page_type = self._get_page_type(
                document_title=document_title,
                page=page,
                include_metadata=include_metadata,
            )

            if page_type == "META":
                continue

            if page_type == "HEADER":
                markdown_parts.extend([f"## {self._format_heading(page.text)}", ""])
                continue

            markdown_parts.extend(self._format_markdown(page.text))
            markdown_parts.append("")
            markdown_parts.extend(page.image_markdown)
            markdown_parts.append("")

        return "\n".join(markdown_parts).strip() + "\n"

    def _get_page_type(
        self,
        document_title: str,
        page: PageContent,
        include_metadata: bool,
    ) -> str:
        if not page.text:
            return "META"

        if include_metadata:
            if self._is_title_only_page(page.text):
                return "HEADER"

            return "CONTENT"

        if self._contains_metadata_markers(page.text):
            return "META"

        if page.number == 1 and self._is_title_only_page(page.text):
            return "META"

        prompt = f"""
Ты фильтруешь страницы для самостоятельного конспекта лекции. Определи тип страницы PDF для итогового Markdown.

Название документа: {document_title}

CONTENT — только факты, аргументы, определения, методы, примеры, результаты или выводы, которые непосредственно объясняют предмет лекции.
HEADER — только тематический заголовок раздела, который непосредственно относится к предмету лекции. Его нужно сохранить как Markdown-заголовок без номера страницы.
META — любая нетематическая информация: титульный слайд, преподаватель или его опыт, программа, карта или план курса, маршрут и цели вебинара, домашнее задание, организационное объявление, вопросы аудитории, благодарность, контакты, ссылки, реклама.
Не включай в конспект карту курса, программу, цели занятия, маршрут вебинара, задания или сведения о преподавателе, даже если они содержат названия тематических разделов. Это всегда META.
Если сомневаешься между CONTENT и META, выбирай META. Если сомневаешься между HEADER и META, выбирай META. Титульная страница с названием документа и авторами всегда META.

Ответь ровно одним словом: CONTENT, HEADER или META.

Текст страницы:
{page.text}
""".strip()
        response = self._chat_model.invoke(prompt)
        decision = str(response.content).strip().upper()

        if decision == "CONTENT" or decision == "HEADER":
            return decision

        return "META"

    def _contains_metadata_markers(
        self,
        text: str,
    ) -> bool:
        normalized_text = text.lower()

        return any(marker in normalized_text for marker in self._metadata_markers)

    def _format_heading(
        self,
        text: str,
    ) -> str:
        return " ".join(self._get_source_lines(text))

    def _is_title_only_page(
        self,
        text: str,
    ) -> bool:
        lines = self._get_source_lines(text)

        if len(lines) > 6:
            return False

        if any(self._is_list_item(line) for line in lines):
            return False

        return len(" ".join(lines)) <= 160 and len(" ".join(lines).split()) <= 25

    def _format_markdown(
        self,
        text: str,
    ) -> list[str]:
        source_lines = self._get_source_lines(text)

        if not source_lines:
            return []

        markdown_lines: list[str] = []
        previous_was_list_item = False

        for line_index, source_line in enumerate(source_lines):
            list_item = self._to_list_item(source_line)

            if list_item:
                markdown_lines.append(list_item)
                previous_was_list_item = True
                continue

            if line_index == 0 and len(source_lines) > 1:
                markdown_lines.extend([f"### {source_line}", ""])
                previous_was_list_item = False
                continue

            if previous_was_list_item:
                markdown_lines.append(f"  {source_line}")
                continue

            if markdown_lines and markdown_lines[-1]:
                markdown_lines[-1] = f"{markdown_lines[-1]} {source_line}"
                continue

            markdown_lines.append(source_line)

        return markdown_lines

    def _get_source_lines(
        self,
        text: str,
    ) -> list[str]:
        text_with_list_breaks = re.sub(
            r"\s*([•●▪◦])\s*",
            r"\n\1 ",
            text,
        )
        source_lines = [
            line.strip()
            for line in text_with_list_breaks.splitlines()
            if line.strip() and not self._is_page_number(line)
        ]

        return source_lines

    def _is_page_number(
        self,
        text: str,
    ) -> bool:
        return re.fullmatch(r"(?:Страница|Page)\s+\d+", text.strip(), re.IGNORECASE) is not None

    def _to_list_item(
        self,
        text: str,
    ) -> str | None:
        bullet_match = re.match(r"^[•●▪◦\-*–—]\s*(.+)$", text)

        if bullet_match:
            return f"- {bullet_match.group(1)}"

        numbered_match = re.match(r"^(\d+[.)])\s*(.+)$", text)

        if numbered_match:
            return f"{numbered_match.group(1)} {numbered_match.group(2)}"

        return None

    def _is_list_item(
        self,
        text: str,
    ) -> bool:
        return self._to_list_item(text) is not None

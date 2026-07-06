import re

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from tutor_bot.application.chat_route import ChatRoute, ChatRouteDecision
from tutor_bot.application.chat_source_query import detect_explicit_chat_source
from tutor_bot.generation.llm_provider import LlmProvider


_ROUTER_PROMPT = """Определи, как обработать запрос пользователя в учебном приложении.
local: пользователь явно спрашивает о своих заметках, локальной базе знаний, изученных материалах или просит найти информацию именно в них.
general: на запрос можно ответить с помощью общих знаний LLM без локальных заметок.
create_note: пользователь просит создать или добавить заметку в активную базу. Извлеки название заметки в note_title. Не придумывай название, если пользователь его не указал.
update_note: пользователь просит дополнить или изменить существующую заметку. Извлеки точное название в note_title и пожелание к изменению в instruction.
start_recall: пользователь просит запустить тест или Active Recall по существующей заметке. Извлеки точное название в note_title.
unavailable: запрос не является вопросом к локальным знаниям или LLM либо требует выполнить недоступное действие.
{format_instructions}"""
_CREATE_NOTE_PATTERN = re.compile(
    r"(?:добав\w*|созда\w*).*?заметк\w*.*?(?:с названием|названи\w*)\s*[\"«](?P<title>.+?)[\"»]",
    re.IGNORECASE,
)
_QUOTED_TITLE_PATTERN = re.compile(r"[\"«](?P<title>.+?)[\"»]")
_CAPABILITIES_MARKERS = (
    "что ты умеешь",
    "что умеет chat",
    "список возможностей",
    "список своего функционала",
    "какие запросы поддерживаешь",
    "какие запросы ты поддерживаешь",
    "поддерживаемые запросы",
)


class ChatSupervisor:
    def __init__(self, provider: LlmProvider) -> None:
        self._provider = provider
        self._parser = PydanticOutputParser(pydantic_object=ChatRouteDecision)

    def select_route(self, question: str) -> ChatRouteDecision:
        deterministic_decision = self._select_deterministic_route(question)

        if deterministic_decision is not None:
            return deterministic_decision

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _ROUTER_PROMPT),
                ("human", "{question}"),
            ]
        )
        messages = prompt.format_messages(
            question=question,
            format_instructions=self._parser.get_format_instructions(),
        )
        response = self._provider.generate(
            messages=[
                {
                    "role": "system" if message.type == "system" else "user",
                    "content": str(message.content),
                }
                for message in messages
            ],
            temperature=0.0,
            max_tokens=200,
            json_schema=ChatRouteDecision.model_json_schema(),
        )

        return self._parser.parse(response.text)

    def _select_deterministic_route(self, question: str) -> ChatRouteDecision | None:
        quoted_title_match = _QUOTED_TITLE_PATTERN.search(question)
        quoted_title = (
            quoted_title_match.group("title").strip()
            if quoted_title_match is not None
            else None
        )
        normalized_question = question.casefold()

        if quoted_title and ("active recall" in normalized_question or "тест" in normalized_question):
            return ChatRouteDecision(
                route=ChatRoute.START_RECALL,
                note_title=quoted_title,
            )

        if quoted_title and any(
            marker in normalized_question
            for marker in ("дополни", "обнови", "измени", "добавь в заметку")
        ):
            return ChatRouteDecision(
                route=ChatRoute.UPDATE_NOTE,
                note_title=quoted_title,
                instruction=question,
            )

        create_note_match = _CREATE_NOTE_PATTERN.search(question)

        if create_note_match is not None:
            return ChatRouteDecision(
                route=ChatRoute.CREATE_NOTE,
                note_title=create_note_match.group("title").strip(),
            )

        if any(marker in normalized_question for marker in _CAPABILITIES_MARKERS):
            return ChatRouteDecision(route=ChatRoute.CAPABILITIES)

        explicit_source = detect_explicit_chat_source(question)

        if explicit_source is not None:
            return ChatRouteDecision(route=ChatRoute(explicit_source))

        return None

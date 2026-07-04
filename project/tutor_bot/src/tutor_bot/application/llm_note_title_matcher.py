import json

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from tutor_bot.application.note_title_match import NoteTitleMatch
from tutor_bot.generation.llm_provider import LlmProvider


_MATCH_PROMPT = """Сопоставь название заметки из запроса пользователя с одним названием из списка.
Учитывай опечатки, разные формы слов и частичное название. Например, запрос Addressables может соответствовать заметке Addressables, AssetBundles, Resources.
Выбирай название только из переданного списка и только если оно заметно ближе остальных вариантов.
Если подходящего варианта нет или несколько вариантов одинаково близки, верни null.
{format_instructions}"""


class LlmNoteTitleMatcher:
    def __init__(self, provider: LlmProvider) -> None:
        self._provider = provider
        self._parser = PydanticOutputParser(pydantic_object=NoteTitleMatch)

    def match(self, requested_title: str, available_titles: list[str]) -> str | None:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _MATCH_PROMPT),
                (
                    "human",
                    "Название из запроса: {requested_title}\nДоступные названия: {available_titles}",
                ),
            ]
        )
        messages = prompt.format_messages(
            requested_title=requested_title,
            available_titles=json.dumps(available_titles, ensure_ascii=False),
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
            json_schema=NoteTitleMatch.model_json_schema(),
        )
        matched_title = self._parser.parse(response.text).matched_title

        if matched_title not in available_titles:
            return None

        return matched_title

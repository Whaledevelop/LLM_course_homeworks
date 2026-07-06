import re
from typing import Literal


_GENERAL_SOURCE_MARKERS = (
    "без локальных заметок",
    "общие знания",
    "знания llm",
    "знания модели",
    "данные из llm",
    "ответ llm",
)
_LOCAL_SOURCE_PATTERN = re.compile(
    r"\b(?:локальн\w*|мо\w*)\s+(?:баз\w+\s+знан\w+|замет\w+)|"
    r"\b(?:в|из|по)\s+(?:мо\w+\s+)?замет\w+|"
    r"\bлокальн\w+\s+баз\w+",
    re.IGNORECASE,
)
_SOURCE_INSTRUCTION_PATTERNS = (
    re.compile(
        r"^\s*(?:использ\w*|опира\w*\s+на|на\s+основе)\s+"
        r"(?:только\s+)?(?:локальн\w+|мо\w+)?\s*"
        r"(?:замет\w+|баз\w+\s+знан\w+)\s*[,.:;\-–—]?\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:использ\w*|опира\w*\s+на|на\s+основе)\s+"
        r"(?:только\s+)?(?:общ\w+\s+знан\w+|знан\w+\s+(?:llm|модел\w+)|"
        r"данн\w+\s+из\s+llm)\s*[,.:;\-–—]?\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s+(?:согласно|по|на\s+основе|использ\w*|опира\w*\s+на)\s+"
        r"(?:только\s+)?(?:локальн\w+|мо\w+)?\s*"
        r"(?:замет\w+|баз\w+\s+знан\w+)\s*[?.!]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s+(?:согласно|по|на\s+основе|использ\w*|опира\w*\s+на)\s+"
        r"(?:только\s+)?(?:общ\w+\s+знан\w+|знан\w+\s+(?:llm|модел\w+)|"
        r"данн\w+\s+из\s+llm)\s*[?.!]*$",
        re.IGNORECASE,
    ),
)


def detect_explicit_chat_source(
    question: str,
) -> Literal["local", "general"] | None:
    normalized_question = question.casefold()

    if any(marker in normalized_question for marker in _GENERAL_SOURCE_MARKERS):
        return "general"

    if _LOCAL_SOURCE_PATTERN.search(question) is not None:
        return "local"

    return None


def strip_chat_source_instruction(question: str) -> str:
    stripped_question = question

    for pattern in _SOURCE_INSTRUCTION_PATTERNS:
        stripped_question = pattern.sub("", stripped_question, count=1)

    normalized_question = stripped_question.strip()

    return normalized_question or question.strip()

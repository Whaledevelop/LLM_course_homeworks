from __future__ import annotations

import re
from dataclasses import dataclass

from dataset import Document


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer: str
    contexts: list[str]
    context_ids: list[str]


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "before",
    "be",
    "because",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "or",
    "should",
    "the",
    "to",
    "what",
    "when",
    "whether",
    "why",
    "with",
}


class LocalRagApplication:
    def __init__(self, documents: list[Document], top_k: int = 2) -> None:
        self._documents = documents
        self._top_k = top_k

    def answer(self, question: str) -> RagAnswer:
        contexts = self.retrieve(question)
        question_tokens = tokenize(question)
        scored_facts = []
        for document in contexts:
            for fact in document.facts:
                fact_tokens = tokenize(fact)
                score = len(question_tokens.intersection(fact_tokens))
                scored_facts.append((score, fact))

        scored_facts.sort(key=lambda item: -item[0])
        selected_facts = [fact for score, fact in scored_facts[:1] if score > 0]
        answer = ". ".join(selected_facts) + "."

        return RagAnswer(
            question=question,
            answer=answer,
            contexts=[document.text for document in contexts],
            context_ids=[document.id for document in contexts],
        )

    def retrieve(self, question: str) -> list[Document]:
        question_tokens = tokenize(question)
        scored_documents = []
        for document in self._documents:
            text_tokens = tokenize(f"{document.title} {document.text}")
            overlap = question_tokens.intersection(text_tokens)
            score = len(overlap) / max(len(question_tokens), 1)
            scored_documents.append((score, document))

        scored_documents.sort(key=lambda item: (-item[0], item[1].id))

        return [document for score, document in scored_documents[: self._top_k] if score > 0]


def tokenize(text: str) -> set[str]:
    tokens = set()
    for token in re.findall(r"[a-zA-Z][a-zA-Z/-]+", text.lower()):
        if token not in STOP_WORDS:
            tokens.add(normalize_token(token))

    return tokens


def normalize_token(token: str) -> str:
    if len(token) > 6 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 5 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]

    return token

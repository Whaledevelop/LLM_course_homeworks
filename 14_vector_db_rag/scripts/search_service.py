from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from chromadb import Collection


@dataclass(frozen=True)
class SearchResult:
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float
    score: float


def semantic_search(
    collection: Collection,
    query: str,
    top_k: int = 5,
    where: dict[str, Any] | None = None,
) -> list[SearchResult]:
    response = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    return _parse_query_response(response)


def hybrid_search(
    collection: Collection,
    query: str,
    top_k: int = 5,
    where: dict[str, Any] | None = None,
    vector_weight: float = 0.5,
) -> list[SearchResult]:
    vector_results = semantic_search(collection, query, top_k=top_k * 4, where=where)
    keyword_results = keyword_search(collection, query, top_k=top_k * 4, where=where)
    scores = defaultdict(float)
    items = {}

    for rank, result in enumerate(vector_results, start=1):
        scores[result.id] += vector_weight / (60 + rank)
        items[result.id] = result

    for rank, result in enumerate(keyword_results, start=1):
        scores[result.id] += (1 - vector_weight) / (60 + rank)
        items[result.id] = result

    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]

    return [
        SearchResult(
            id=result_id,
            text=items[result_id].text,
            metadata=items[result_id].metadata,
            distance=items[result_id].distance,
            score=scores[result_id],
        )
        for result_id in ranked_ids
    ]


def keyword_search(
    collection: Collection,
    query: str,
    top_k: int = 5,
    where: dict[str, Any] | None = None,
) -> list[SearchResult]:
    response = collection.get(where=where, include=["documents", "metadatas"])
    documents = response["documents"]
    ids = response["ids"]
    metadatas = response["metadatas"]
    query_terms = _tokens(query)
    document_terms = [_tokens(document) for document in documents]
    document_count = len(documents)
    document_frequency = Counter()

    for terms in document_terms:
        document_frequency.update(set(terms))

    scored_results = []

    for document_id, document, metadata, terms in zip(ids, documents, metadatas, document_terms):
        term_counts = Counter(terms)
        score = 0.0

        for term in query_terms:
            frequency = term_counts[term]
            if frequency == 0:
                continue

            inverse_document_frequency = math.log((document_count + 1) / (document_frequency[term] + 1)) + 1
            score += frequency * inverse_document_frequency

        if score > 0:
            scored_results.append(
                SearchResult(
                    id=document_id,
                    text=document,
                    metadata=metadata,
                    distance=0.0,
                    score=score,
                )
            )

    return sorted(scored_results, key=lambda result: result.score, reverse=True)[:top_k]


def _parse_query_response(response: dict[str, Any]) -> list[SearchResult]:
    results = []

    for document_id, document, metadata, distance in zip(
        response["ids"][0],
        response["documents"][0],
        response["metadatas"][0],
        response["distances"][0],
    ):
        results.append(
            SearchResult(
                id=document_id,
                text=document,
                metadata=metadata,
                distance=distance,
                score=1 / (1 + distance),
            )
        )

    return results


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zа-яё0-9]+", text.lower())

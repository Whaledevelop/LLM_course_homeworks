from __future__ import annotations

from chromadb import Collection

from search_service import SearchResult, hybrid_search, semantic_search


def answer_question(
    collection: Collection,
    question: str,
    top_k: int = 4,
    where: dict[str, object] | None = None,
    use_hybrid: bool = True,
) -> tuple[str, list[SearchResult]]:
    results = (
        hybrid_search(collection, question, top_k=top_k, where=where)
        if use_hybrid
        else semantic_search(collection, question, top_k=top_k, where=where)
    )
    context = "\n".join(
        f"{index}. {result.text} Source: {result.metadata['source']}, title: {result.metadata['title']}."
        for index, result in enumerate(results, start=1)
    )
    answer = (
        f"Answer based on retrieved context:\n{context}\n\n"
        f"Short conclusion: the most relevant evidence for '{question}' is in "
        f"{', '.join(result.metadata['title'] for result in results[:2])}."
    )

    return answer, results

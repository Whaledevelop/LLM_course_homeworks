from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    title: str
    text: str
    category: str
    source: str
    year: int
    difficulty: str


TOPICS = {
    "vector_database": [
        "Chroma stores vectors with metadata and supports local persistent collections.",
        "Qdrant focuses on payload filtering and production friendly vector search.",
        "Milvus is designed for distributed vector retrieval at very large scale.",
        "Pinecone provides managed vector indexes without operating database nodes.",
    ],
    "ann_algorithm": [
        "HNSW uses a navigable graph for approximate nearest neighbor vector search and usually gives high recall with low latency.",
        "IVF partitions vectors into clusters and searches only selected inverted lists for approximate nearest neighbor retrieval.",
        "ANNOY builds random projection trees for approximate nearest neighbor search and is simple for mostly static indexes.",
        "Product quantization compresses vector embeddings and trades a small accuracy loss for memory savings.",
    ],
    "rag_pipeline": [
        "RAG indexes documents, retrieves relevant chunks, and sends context to a language model.",
        "Chunk size and overlap affect whether retrieval returns complete facts.",
        "Grounded answers should cite retrieved sources and avoid unsupported facts.",
        "Context compression and reranking can improve the final evidence passed to generation.",
    ],
    "metadata_filtering": [
        "Metadata filters restrict vector search to a selected category, source, year, or tenant.",
        "Payload indexes make filtered search faster when the collection contains many documents.",
        "A RAG system can combine semantic similarity with structured business constraints.",
        "Filtering before generation reduces irrelevant context and lowers hallucination risk.",
    ],
    "evaluation": [
        "Recall at k measures whether relevant documents appear among the retrieved results.",
        "MRR rewards systems that place the first relevant document closer to the top.",
        "Latency should be measured together with retrieval quality because ANN parameters change both.",
        "LLM as a judge can estimate faithfulness when exact reference answers are unavailable.",
    ],
    "hybrid_search": [
        "Hybrid search combines vector similarity with lexical scoring such as BM25.",
        "Reciprocal rank fusion merges ranked lists without requiring normalized scores.",
        "Keyword search helps with exact names, identifiers, versions, and rare technical terms.",
        "Semantic retrieval helps when the query uses synonyms or paraphrases.",
    ],
}

DIFFICULTIES = ["basic", "intermediate", "advanced"]
SOURCES = ["lecture_notes", "documentation", "experiment_log"]


def generate_dataset(path: Path, documents_per_category: int = 200) -> list[DocumentRecord]:
    records = []

    for category, sentences in TOPICS.items():
        for index in range(documents_per_category):
            first_sentence = sentences[index % len(sentences)]
            second_sentence = sentences[(index + 1) % len(sentences)]
            difficulty = DIFFICULTIES[index % len(DIFFICULTIES)]
            source = SOURCES[index % len(SOURCES)]
            year = 2023 + index % 4
            title = f"{category.replace('_', ' ').title()} note {index + 1}"
            text = (
                f"{first_sentence} {second_sentence} "
                f"This note belongs to the {category} topic, has {difficulty} difficulty, "
                f"and is useful for retrieval augmented generation experiments."
            )
            record = DocumentRecord(
                id=f"{category}-{index + 1:04d}",
                title=title,
                text=text,
                category=category,
                source=source,
                year=year,
                difficulty=difficulty,
            )
            records.append(record)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    return records


def load_dataset(path: Path) -> list[DocumentRecord]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            payload = json.loads(line)
            records.append(DocumentRecord(**payload))

    return records

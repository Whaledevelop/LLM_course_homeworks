from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from dataset import DocumentRecord
from embedding_service import HashingEmbeddingFunction
from index_service import create_client, index_documents, recreate_collection
from search_service import hybrid_search, semantic_search


@dataclass(frozen=True)
class EvaluationQuery:
    query: str
    category: str


EVALUATION_QUERIES = [
    EvaluationQuery("graph based nearest neighbor search with high recall", "ann_algorithm"),
    EvaluationQuery("IVF partitions vectors into clusters selected inverted lists", "ann_algorithm"),
    EvaluationQuery("RAG indexes documents retrieves relevant chunks sends context language model", "rag_pipeline"),
    EvaluationQuery("restrict search by source year and category metadata", "metadata_filtering"),
    EvaluationQuery("combine semantic retrieval with keyword ranking", "hybrid_search"),
    EvaluationQuery("Recall at k relevant documents retrieved results MRR first relevant", "evaluation"),
    EvaluationQuery("local persistent vector database for prototypes", "vector_database"),
]


def evaluate_collection(collection, top_k: int = 5, use_hybrid: bool = False) -> dict[str, float]:
    latencies = []
    recalls = []
    reciprocal_ranks = []

    for evaluation_query in EVALUATION_QUERIES:
        started_at = time.perf_counter()
        results = (
            hybrid_search(collection, evaluation_query.query, top_k=top_k)
            if use_hybrid
            else semantic_search(collection, evaluation_query.query, top_k=top_k)
        )
        latencies.append((time.perf_counter() - started_at) * 1000)
        relevant_ranks = [
            rank
            for rank, result in enumerate(results, start=1)
            if result.metadata["category"] == evaluation_query.category
        ]
        recalls.append(1.0 if relevant_ranks else 0.0)
        reciprocal_ranks.append(1 / relevant_ranks[0] if relevant_ranks else 0.0)

    return {
        "top_k": top_k,
        "recall_at_k": statistics.mean(recalls),
        "mrr": statistics.mean(reciprocal_ranks),
        "latency_ms_mean": statistics.mean(latencies),
        "latency_ms_p95": sorted(latencies)[int(len(latencies) * 0.95) - 1],
    }


def benchmark_profiles(
    database_path: Path,
    records: list[DocumentRecord],
    embedding_function: HashingEmbeddingFunction,
) -> list[dict[str, object]]:
    client = create_client(database_path)
    profiles = [
        {"name": "cosine_fast", "space": "cosine", "search_ef": 20, "max_neighbors": 12},
        {"name": "cosine_balanced", "space": "cosine", "search_ef": 50, "max_neighbors": 16},
        {"name": "cosine_accurate", "space": "cosine", "search_ef": 100, "max_neighbors": 32},
        {"name": "l2_balanced", "space": "l2", "search_ef": 50, "max_neighbors": 16},
        {"name": "ip_balanced", "space": "ip", "search_ef": 50, "max_neighbors": 16},
    ]
    rows = []

    for profile in profiles:
        collection = recreate_collection(
            client=client,
            name=profile["name"],
            embedding_function=embedding_function,
            space=profile["space"],
            search_ef=profile["search_ef"],
            max_neighbors=profile["max_neighbors"],
        )
        started_at = time.perf_counter()
        index_documents(collection, records)
        index_time_ms = (time.perf_counter() - started_at) * 1000
        semantic_metrics = evaluate_collection(collection, top_k=5, use_hybrid=False)
        hybrid_metrics = evaluate_collection(collection, top_k=5, use_hybrid=True)
        rows.append(
            {
                **profile,
                "documents": len(records),
                "index_time_ms": index_time_ms,
                "semantic_recall_at_5": semantic_metrics["recall_at_k"],
                "semantic_mrr": semantic_metrics["mrr"],
                "semantic_latency_ms": semantic_metrics["latency_ms_mean"],
                "hybrid_recall_at_5": hybrid_metrics["recall_at_k"],
                "hybrid_mrr": hybrid_metrics["mrr"],
                "hybrid_latency_ms": hybrid_metrics["latency_ms_mean"],
            }
        )

    return rows

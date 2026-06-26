from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if sys.version_info >= (3, 14):
    raise RuntimeError(
        "Use Python 3.10-3.13 for this homework. "
        "ChromaDB dependencies include compiled wheels that are not compatible with this Python 3.14 environment."
    )

from dataset import generate_dataset, load_dataset
from embedding_service import HashingEmbeddingFunction
from evaluation import benchmark_profiles, evaluate_collection
from index_service import create_client, index_documents, recreate_collection
from rag_service import answer_question
from search_service import hybrid_search, semantic_search


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "documents.jsonl"
DATABASE_PATH = ROOT / "data" / "chroma"
REPORT_PATH = ROOT / "data" / "benchmark_results.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--query", default="HNSW navigable graph approximate nearest neighbor high recall low latency")
    args = parser.parse_args()

    if args.rebuild or not DATASET_PATH.exists():
        records = generate_dataset(DATASET_PATH, documents_per_category=200)
    else:
        records = load_dataset(DATASET_PATH)

    embedding_function = HashingEmbeddingFunction(dimensions=384)
    client = create_client(DATABASE_PATH)
    collection = recreate_collection(
        client=client,
        name="rag_homework_cosine",
        embedding_function=embedding_function,
        space="cosine",
        construction_ef=100,
        search_ef=50,
        max_neighbors=16,
    )
    index_documents(collection, records)

    print(f"Indexed documents: {collection.count()}")
    print("\nSemantic search:")
    for result in semantic_search(collection, args.query, top_k=3):
        print(f"- {result.id} | {result.metadata['category']} | distance={result.distance:.4f}")

    print("\nFiltered search, category=ann_algorithm:")
    for result in semantic_search(collection, args.query, top_k=3, where={"category": "ann_algorithm"}):
        print(f"- {result.id} | {result.metadata['title']} | distance={result.distance:.4f}")

    print("\nHybrid search:")
    for result in hybrid_search(collection, args.query, top_k=3):
        print(f"- {result.id} | {result.metadata['category']} | score={result.score:.4f}")

    answer, sources = answer_question(collection, args.query, top_k=4)
    print("\nRAG answer:")
    print(answer)
    print("\nSource ids:")
    print(", ".join(result.id for result in sources))

    semantic_metrics = evaluate_collection(collection, top_k=5, use_hybrid=False)
    hybrid_metrics = evaluate_collection(collection, top_k=5, use_hybrid=True)
    print("\nSemantic retrieval metrics:")
    for key, value in semantic_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nHybrid retrieval metrics:")
    for key, value in hybrid_metrics.items():
        print(f"{key}: {value:.4f}")

    if args.benchmark:
        benchmark = benchmark_profiles(DATABASE_PATH, records, embedding_function)
        with REPORT_PATH.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(benchmark[0].keys()))
            writer.writeheader()
            writer.writerows(benchmark)

        print(f"\nBenchmark saved to {REPORT_PATH}")
        for row in benchmark:
            print(row)


if __name__ == "__main__":
    main()

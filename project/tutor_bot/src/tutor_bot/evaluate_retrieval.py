import argparse

from tutor_bot.config import get_settings
from tutor_bot.evaluation.golden_loader import load_golden_dataset
from tutor_bot.evaluation.retrieval_evaluator import RetrievalEvaluator
from tutor_bot.indexing.bm25_chunk_index import Bm25ChunkIndex
from tutor_bot.indexing.chroma_chunk_index import ChromaChunkIndex
from tutor_bot.indexing.sentence_transformer_embedding_service import (
    SentenceTransformerEmbeddingService,
)
from tutor_bot.retrieval.hybrid_search_service import HybridSearchService
from tutor_bot.retrieval.reranker_context_gate import RerankerContextGate
from tutor_bot.retrieval.sentence_transformer_reranker import SentenceTransformerReranker


def main() -> int:
    arguments = _parse_arguments()
    settings = get_settings()

    search_service = HybridSearchService(
        SentenceTransformerEmbeddingService(device="cpu"),
        ChromaChunkIndex(settings.indexes_dir / "chroma"),
        Bm25ChunkIndex(settings.indexes_dir / "bm25"),
        SentenceTransformerReranker(device="cpu"),
    )

    context_gate = RerankerContextGate(
        minimum_reranker_score=arguments.minimum_reranker_score,
        context_limit=arguments.context_limit,
    )

    evaluator = RetrievalEvaluator(
        search_service,
        context_gate,
        retrieval_limit=arguments.retrieval_limit,
        recall_k=arguments.recall_k,
    )

    dataset = load_golden_dataset(settings.evaluation_dir / "goldens.json")
    report = evaluator.evaluate(dataset)

    print(report.model_dump_json(indent=2))

    return 0


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--minimum-reranker-score",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--context-limit",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--retrieval-limit",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--recall-k",
        type=int,
        default=5,
    )

    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

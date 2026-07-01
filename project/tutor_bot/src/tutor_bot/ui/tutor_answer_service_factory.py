import streamlit as st

from tutor_bot.application.tutor_answer_service import TutorAnswerService
from tutor_bot.config import get_settings
from tutor_bot.generation.ollama_grounded_answer_generator import OllamaGroundedAnswerGenerator
from tutor_bot.indexing.bm25_chunk_index import Bm25ChunkIndex
from tutor_bot.indexing.chroma_chunk_index import ChromaChunkIndex
from tutor_bot.indexing.sentence_transformer_embedding_service import (
    SentenceTransformerEmbeddingService,
)
from tutor_bot.retrieval.hybrid_search_service import HybridSearchService
from tutor_bot.retrieval.reranker_context_gate import RerankerContextGate
from tutor_bot.retrieval.sentence_transformer_reranker import SentenceTransformerReranker


@st.cache_resource(show_spinner="Загрузка моделей поиска...")
def create_tutor_answer_service() -> TutorAnswerService:
    settings = get_settings()

    search_service = HybridSearchService(
        SentenceTransformerEmbeddingService(device="cpu"),
        ChromaChunkIndex(settings.indexes_dir / "chroma"),
        Bm25ChunkIndex(settings.indexes_dir / "bm25"),
        SentenceTransformerReranker(device="cpu"),
    )

    return TutorAnswerService(
        search_service,
        RerankerContextGate(
            minimum_reranker_score=0.0,
            context_limit=5,
        ),
        OllamaGroundedAnswerGenerator(
            settings.ollama_base_url,
            model_name=settings.ollama_model,
            think=settings.ollama_think,
        ),
    )

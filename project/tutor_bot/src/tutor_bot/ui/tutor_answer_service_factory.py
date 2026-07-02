import streamlit as st

from tutor_bot.application.active_recall_service import ActiveRecallService
from tutor_bot.application.assignment_review_service import AssignmentReviewService
from tutor_bot.application.note_command_service import NoteCommandService
from tutor_bot.application.note_query_service import NoteQueryService
from tutor_bot.application.observability_event_service import ObservabilityEventService
from tutor_bot.application.tutor_answer_service import TutorAnswerService
from tutor_bot.config import get_settings
from tutor_bot.generation.ollama_grounded_assignment_reviewer import (
    OllamaGroundedAssignmentReviewer,
)
from tutor_bot.generation.ollama_grounded_answer_generator import OllamaGroundedAnswerGenerator
from tutor_bot.generation.ollama_grounded_recall_answer_reviewer import (
    OllamaGroundedRecallAnswerReviewer,
)
from tutor_bot.generation.ollama_grounded_recall_exercise_generator import (
    OllamaGroundedRecallExerciseGenerator,
)
from tutor_bot.generation.ollama_note_metadata_suggester import (
    OllamaNoteMetadataSuggester,
)
from tutor_bot.indexing.bm25_chunk_index import Bm25ChunkIndex
from tutor_bot.indexing.chroma_chunk_index import ChromaChunkIndex
from tutor_bot.indexing.sentence_transformer_embedding_service import (
    SentenceTransformerEmbeddingService,
)
from tutor_bot.infrastructure.active_recall_history_repository import (
    ActiveRecallHistoryRepository,
)
from tutor_bot.infrastructure.jsonl_observability_event_repository import (
    JsonlObservabilityEventRepository,
)
from tutor_bot.retrieval.hybrid_search_service import HybridSearchService
from tutor_bot.retrieval.reranker_context_gate import RerankerContextGate
from tutor_bot.retrieval.sentence_transformer_reranker import SentenceTransformerReranker


@st.cache_resource(show_spinner="Загрузка моделей поиска...")
def create_hybrid_search_service() -> HybridSearchService:
    settings = get_settings()

    return HybridSearchService(
        SentenceTransformerEmbeddingService(device="cpu"),
        ChromaChunkIndex(settings.indexes_dir / "chroma"),
        Bm25ChunkIndex(settings.indexes_dir / "bm25"),
        SentenceTransformerReranker(device="cpu"),
    )


@st.cache_resource
def create_observability_event_service() -> ObservabilityEventService:
    settings = get_settings()

    return ObservabilityEventService(
        JsonlObservabilityEventRepository(
            settings.history_dir / "observability_events.jsonl",
        ),
    )


@st.cache_resource
def create_tutor_answer_service() -> TutorAnswerService:
    settings = get_settings()

    return TutorAnswerService(
        create_hybrid_search_service(),
        RerankerContextGate(
            minimum_reranker_score=0.0,
            context_limit=5,
        ),
        OllamaGroundedAnswerGenerator(
            settings.ollama_base_url,
            model_name=settings.ollama_model,
            think=settings.ollama_think,
        ),
        observability_event_service=create_observability_event_service(),
    )


@st.cache_resource
def create_assignment_review_service() -> AssignmentReviewService:
    settings = get_settings()

    return AssignmentReviewService(
        create_hybrid_search_service(),
        RerankerContextGate(
            minimum_reranker_score=0.0,
            context_limit=5,
        ),
        OllamaGroundedAssignmentReviewer(
            settings.ollama_base_url,
            model_name=settings.ollama_model,
            think=settings.ollama_think,
        ),
        observability_event_service=create_observability_event_service(),
    )


@st.cache_resource
def create_active_recall_service(
    _note_query_service: NoteQueryService,
    _note_command_service: NoteCommandService | None = None,
) -> ActiveRecallService:
    settings = get_settings()

    return ActiveRecallService(
        _note_query_service,
        OllamaGroundedRecallExerciseGenerator(
            settings.ollama_base_url,
            model_name=settings.ollama_model,
            think=settings.ollama_think,
        ),
        OllamaGroundedRecallAnswerReviewer(
            settings.ollama_base_url,
            model_name=settings.ollama_model,
            think=settings.ollama_think,
        ),
        ActiveRecallHistoryRepository(
            settings.history_dir / "active_recall.jsonl",
        ),
        note_command_service=_note_command_service,
        observability_event_service=create_observability_event_service(),
    )


@st.cache_resource
def create_note_metadata_suggester() -> OllamaNoteMetadataSuggester:
    settings = get_settings()

    return OllamaNoteMetadataSuggester(
        settings.ollama_base_url,
        model_name=settings.ollama_model,
        think=settings.ollama_think,
        observability_event_service=create_observability_event_service(),
    )

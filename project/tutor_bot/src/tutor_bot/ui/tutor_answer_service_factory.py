import streamlit as st
from pathlib import Path

from tutor_bot.application.active_recall_service import ActiveRecallService
from tutor_bot.application.assignment_review_service import AssignmentReviewService
from tutor_bot.application.chat_service import ChatService
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
from tutor_bot.generation.llm_note_content_generator import LlmNoteContentGenerator
from tutor_bot.generation.focused_recall_exercise_generator import FocusedRecallExerciseGenerator
from tutor_bot.generation.vacancy_analyzer import VacancyAnalyzer
from tutor_bot.generation.llm_provider import LlmProvider
from tutor_bot.generation.observed_llm_provider import ObservedLlmProvider
from tutor_bot.generation.ollama_provider import OllamaProvider
from tutor_bot.generation.yandex_provider import YandexProvider
from tutor_bot.indexing.bm25_chunk_index import Bm25ChunkIndex
from tutor_bot.indexing.chroma_chunk_index import ChromaChunkIndex
from tutor_bot.indexing.corpus_chunk_builder import CorpusChunkBuilder
from tutor_bot.indexing.full_reindex_service import FullReindexService
from tutor_bot.indexing.markdown_cleaner import MarkdownCleaner
from tutor_bot.indexing.markdown_section_chunker import MarkdownSectionChunker
from tutor_bot.indexing.markdown_section_splitter import MarkdownSectionSplitter
from tutor_bot.indexing.note_chunk_builder import NoteChunkBuilder
from tutor_bot.indexing.sentence_transformer_embedding_service import (
    SentenceTransformerEmbeddingService,
)
from tutor_bot.infrastructure.active_recall_history_repository import (
    ActiveRecallHistoryRepository,
)
from tutor_bot.infrastructure.database_notes_repository import DatabaseNotesRepository
from tutor_bot.infrastructure.langfuse_observability_event_repository import (
    LangfuseObservabilityEventRepository,
)
from tutor_bot.retrieval.hybrid_search_service import HybridSearchService
from tutor_bot.retrieval.reranker_context_gate import RerankerContextGate
from tutor_bot.retrieval.sentence_transformer_reranker import SentenceTransformerReranker
from tutor_bot.application.vacancy_matching_service import VacancyMatchingService
from tutor_bot.application.vacancy_preparation_service import VacancyPreparationService
from tutor_bot.ui.llm_session_state import get_active_model, get_active_provider, record_usage


@st.cache_resource(show_spinner="Загрузка моделей поиска...")
def create_hybrid_search_service(db_id: str) -> HybridSearchService:
    settings = get_settings()
    indexes_dir = settings.indexes_dir / db_id

    return HybridSearchService(
        SentenceTransformerEmbeddingService(device="cpu"),
        ChromaChunkIndex(indexes_dir / "chroma"),
        Bm25ChunkIndex(indexes_dir / "bm25"),
        SentenceTransformerReranker(device="cpu"),
    )


@st.cache_resource
def create_observability_event_service(
    provider_name: str | None = None,
    model_name: str | None = None,
) -> ObservabilityEventService:
    settings = get_settings()

    return ObservabilityEventService(
        LangfuseObservabilityEventRepository(
            settings.langfuse_public_key,
            settings.langfuse_secret_key,
            settings.langfuse_base_url,
        ),
        generation_provider=provider_name,
        generation_model=model_name,
    )


def create_tutor_answer_service(db_id: str) -> TutorAnswerService:
    provider = _create_llm_provider()

    return TutorAnswerService(
        create_hybrid_search_service(db_id),
        RerankerContextGate(
            minimum_reranker_score=0.0,
            context_limit=5,
        ),
        OllamaGroundedAnswerGenerator(provider),
        observability_event_service=_get_provider_observability_service(provider),
    )


def create_chat_service(
    db_id: str,
    note_query_service: NoteQueryService,
    note_command_service: NoteCommandService,
) -> ChatService:
    return ChatService(
        _create_llm_provider(),
        lambda: create_tutor_answer_service(db_id),
        note_command_service,
        create_note_content_generator(),
        note_query_service,
        create_active_recall_service(
            note_query_service,
            note_command_service,
            db_id,
        ),
    )


def create_assignment_review_service(db_id: str) -> AssignmentReviewService:
    provider = _create_llm_provider()

    return AssignmentReviewService(
        create_hybrid_search_service(db_id),
        RerankerContextGate(
            minimum_reranker_score=0.0,
            context_limit=5,
        ),
        OllamaGroundedAssignmentReviewer(provider),
        observability_event_service=_get_provider_observability_service(provider),
    )


def create_active_recall_service(
    _note_query_service: NoteQueryService,
    _note_command_service: NoteCommandService | None = None,
    db_id: str = "default",
) -> ActiveRecallService:
    exercise_provider = _create_llm_provider()
    review_provider = _create_llm_provider()

    return ActiveRecallService(
        _note_query_service,
        OllamaGroundedRecallExerciseGenerator(exercise_provider),
        OllamaGroundedRecallAnswerReviewer(review_provider),
        ActiveRecallHistoryRepository(
            get_settings().history_dir / db_id / "active_recall.jsonl",
        ),
        note_command_service=_note_command_service,
        observability_event_service=_get_provider_observability_service(exercise_provider),
    )


def create_note_metadata_suggester() -> OllamaNoteMetadataSuggester:
    provider = _create_llm_provider()

    return OllamaNoteMetadataSuggester(
        provider,
    )


def create_note_content_generator() -> LlmNoteContentGenerator:
    return LlmNoteContentGenerator(_create_llm_provider())


def create_vacancy_analyzer() -> VacancyAnalyzer:
    return VacancyAnalyzer(_create_llm_provider())


def create_vacancy_matching_service(db_id: str) -> VacancyMatchingService:
    return VacancyMatchingService(
        create_hybrid_search_service(db_id),
        _create_llm_provider(),
    )


def create_vacancy_preparation_service(
    note_query_service: NoteQueryService,
    note_command_service: NoteCommandService,
    db_id: str,
) -> VacancyPreparationService:
    provider = _create_llm_provider()

    return VacancyPreparationService(
        note_query_service,
        FocusedRecallExerciseGenerator(provider),
        create_active_recall_service(
            note_query_service,
            note_command_service,
            db_id,
        ),
    )


def rebuild_database_search_index(db_id: str, root_path: str) -> int:
    settings = get_settings()
    indexes_dir = settings.indexes_dir / db_id
    metadata_repository = DatabaseNotesRepository(
        settings.data_dir / "metadata",
        db_id,
        Path(root_path),
    )
    note_chunk_builder = NoteChunkBuilder(
        MarkdownCleaner(),
        MarkdownSectionSplitter(),
        MarkdownSectionChunker(),
    )
    service = FullReindexService(
        CorpusChunkBuilder(metadata_repository, Path(root_path), note_chunk_builder),
        SentenceTransformerEmbeddingService(device="cpu"),
        ChromaChunkIndex(indexes_dir / "chroma"),
        Bm25ChunkIndex(indexes_dir / "bm25"),
    )

    return service.rebuild()


def _create_llm_provider() -> LlmProvider:
    settings = get_settings()
    provider_name = get_active_provider(settings.llm_provider)

    if provider_name == "yandex":
        model_name = get_active_model(settings.yandex_model)

        provider = YandexProvider(
            settings.yandex_base_url,
            settings.yandex_api_key,
            settings.yandex_folder_id,
            model_name,
            max_tokens=settings.yandex_max_tokens,
            temperature=settings.yandex_temperature,
            usage_callback=record_usage,
        )
    else:
        model_name = get_active_model(settings.ollama_model)
        provider = OllamaProvider(
            settings.ollama_base_url,
            model_name,
            think=settings.ollama_think,
            usage_callback=record_usage,
        )

    return ObservedLlmProvider(
        provider,
        create_observability_event_service(
            provider.provider_name,
            provider.model_name,
        ),
    )


def _get_provider_observability_service(
    provider: LlmProvider,
) -> ObservabilityEventService:
    return create_observability_event_service(
        provider.provider_name,
        provider.model_name,
    )

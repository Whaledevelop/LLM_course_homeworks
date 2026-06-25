from dataclasses import dataclass
from hashlib import sha256

from langfuse import Langfuse
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from document_service import get_source_files, get_source_name
from llm_service import HuggingFaceInferenceEmbeddings, generate_chat_response
from observability import get_langfuse_client
from settings import Settings


@dataclass(frozen=True)
class Source:
    content: str
    heading: str
    score: float


@dataclass(frozen=True)
class Answer:
    content: str
    sources: list[Source]


def create_index(settings: Settings) -> int:
    if not get_source_files(settings):
        raise RuntimeError("Документы не добавлены. Выберите файлы или папку справа.")

    if settings.chroma_directory.exists():
        vector_store = _vector_store(settings)
        chunk_count = vector_store._collection.count()
        if chunk_count > 0:
            return chunk_count

    documents = _load_chunks(settings)
    settings.chroma_directory.mkdir(parents=True, exist_ok=True)
    Chroma.from_documents(
        documents=documents,
        embedding=_embeddings(settings),
        collection_name=_collection_name(settings),
        persist_directory=str(settings.chroma_directory),
    )

    chunk_count = len(documents)

    return chunk_count


def index_exists(settings: Settings) -> bool:
    if not get_source_files(settings) or not settings.chroma_directory.exists():
        return False

    vector_store = _vector_store(settings)

    return vector_store._collection.count() > 0


def answer_question(question: str, settings: Settings) -> Answer:
    if not get_source_files(settings):
        raise RuntimeError("Документы не добавлены. Выберите файлы или папку справа.")

    if not index_exists(settings):
        raise RuntimeError("Индекс не готов. Нажмите «Сохранить и обновить индекс» справа.")

    langfuse_client = get_langfuse_client(settings)
    if langfuse_client is None:
        answer = _answer_question(question, settings, None)

        return answer

    with langfuse_client.start_as_current_observation(
        name="answer_question",
        as_type="chain",
        input={"question": question},
    ) as answer_observation:
        answer = _answer_question(question, settings, langfuse_client)
        answer_observation.update(output={"answer": answer.content})

    langfuse_client.flush()

    return answer


def _answer_question(question: str, settings: Settings, langfuse_client: Langfuse | None) -> Answer:
    if langfuse_client is None:
        sources = _search_sources(question, settings)
        if not sources:
            return Answer(
                content="В материалах урока нет информации для ответа на этот вопрос.",
                sources=[],
            )

        response = _generate_answer(question, sources, settings)

        return Answer(content=response.content, sources=sources)

    with langfuse_client.start_as_current_observation(
        name="retrieve_context",
        as_type="retriever",
        input={"question": question},
    ) as retrieval_observation:
        sources = _search_sources(question, settings)
        retrieval_observation.update(
            output={
                "document_count": len(sources),
                "sources": _sources_payload(sources),
            }
        )

    if not sources:
        return Answer(
            content="В материалах урока нет информации для ответа на этот вопрос.",
            sources=[],
        )

    with langfuse_client.start_as_current_observation(
        name="generate_answer",
        as_type="generation",
        input={"question": question, "context": _build_context(sources)},
        model=settings.chat_model,
        model_parameters={"temperature": 0},
        cost_details={"input": 0, "output": 0, "total": 0},
    ) as generation_observation:
        response = _generate_answer(question, sources, settings)
        generation_observation.update(
            output=response.content,
            usage_details=response.usage_details,
        )

    return Answer(content=response.content, sources=sources)


def _search_sources(question: str, settings: Settings) -> list[Source]:
    vector_store = _vector_store(settings)
    search_results = vector_store.similarity_search_with_score(question, k=4)

    sources = [
        Source(content=document.page_content, heading=document.metadata["heading"], score=_distance_to_score(distance))
        for document, distance in search_results
    ]

    return sources


def _distance_to_score(distance: float) -> float:
    score = 1 / (1 + max(distance, 0))

    return score


def _generate_answer(question: str, sources: list[Source], settings: Settings):
    context = _build_context(sources)
    system_message = """
    Ты справочник по загруженным пользователем документам.
    
    Правила ответа:
    1. Отвечай только на русском языке.
    2. Используй только информацию из контекста.
    3. Не добавляй факты, выводы, оценки и объяснения, которых нет в контексте.
    4. Не используй слова вроде «масштабируемый», «эффективный», «удобный», если этого нет в контексте.
    5. Формулируй ответ максимально близко к тексту заметок.
    6. Если информации недостаточно, ответь строго:
    «В материалах урока нет информации для ответа на этот вопрос.»
    7. Не упоминай источники внутри ответа — они показываются отдельно.
    """.strip()
    user_message = f"""
    Контекст:
    {context}
    
    Вопрос:
    {question}
    
    Ответь кратко и строго по контексту.
    """.strip()

    response = generate_chat_response(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        settings=settings,
        max_tokens=700,
    )

    return response


def _build_context(sources: list[Source]) -> str:
    context = "\n\n".join(
        f"[{source.heading}]\n{source.content}"
        for source in sources
    )

    return context


def _sources_payload(sources: list[Source]) -> list[dict]:
    payload = [
        {"heading": source.heading, "content": source.content, "score": source.score}
        for source in sources
    ]

    return payload


def _load_chunks(settings: Settings) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section"), ("###", "subsection")],
        strip_headers=False,
    )
    chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
    )
    chunks = []
    for source_file in get_source_files(settings):
        source_text = source_file.read_text(encoding="utf-8")
        source_name = get_source_name(source_file, settings)
        if source_file.suffix.lower() == ".md":
            sections = header_splitter.split_text(source_text)
        else:
            sections = [Document(page_content=source_text, metadata={"title": source_file.stem})]

        document_chunks = chunk_splitter.split_documents(sections)
        for chunk in document_chunks:
            chunk.metadata["source_file"] = source_name

        chunks.extend(document_chunks)

    for chunk_index, chunk in enumerate(chunks):
        heading = chunk.metadata.get("subsection") or chunk.metadata.get("section") or chunk.metadata.get("title") or "Без заголовка"
        source_file = chunk.metadata["source_file"]
        chunk_id = sha256(f"{source_file}:{chunk_index}:{chunk.page_content}".encode("utf-8")).hexdigest()[:12]
        chunk.metadata = {
            "chunk_id": chunk_id,
            "heading": f"{source_file}: {heading}",
            "source_file": source_file,
        }

    return chunks


def _embeddings(settings: Settings) -> HuggingFaceInferenceEmbeddings:
    return HuggingFaceInferenceEmbeddings(settings)


def _vector_store(settings: Settings) -> Chroma:
    return Chroma(
        collection_name=_collection_name(settings),
        embedding_function=_embeddings(settings),
        persist_directory=str(settings.chroma_directory),
    )


def _collection_name(settings: Settings) -> str:
    content_hash = sha256()
    content_hash.update(settings.embedding_model.encode("utf-8"))
    for source_file in get_source_files(settings):
        source_name = get_source_name(source_file, settings)
        content_hash.update(source_name.encode("utf-8"))
        content_hash.update(source_file.read_bytes())

    collection_name = f"knowledge_{content_hash.hexdigest()[:16]}"

    return collection_name

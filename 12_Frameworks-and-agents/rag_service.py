from dataclasses import dataclass
from hashlib import sha256
import shutil

from langfuse import Langfuse
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

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
    documents = _load_chunks(settings)
    if settings.chroma_directory.exists():
        shutil.rmtree(settings.chroma_directory)

    settings.chroma_directory.mkdir(parents=True, exist_ok=True)
    Chroma.from_documents(
        documents=documents,
        embedding=_embeddings(settings),
        collection_name="lesson_12",
        persist_directory=str(settings.chroma_directory),
    )

    return len(documents)


def index_exists(settings: Settings) -> bool:
    if not settings.chroma_directory.exists():
        return False

    vector_store = _vector_store(settings)

    return vector_store._collection.count() > 0


def answer_question(question: str, settings: Settings) -> Answer:
    if not index_exists(settings):
        raise RuntimeError("Индекс не создан. Нажмите «Создать индекс».")

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

        return Answer(content=str(response.content), sources=sources)

    with langfuse_client.start_as_current_observation(
        name="retrieve_context",
        as_type="retriever",
        input={"question": question},
    ) as retrieval_observation:
        sources = _search_sources(question, settings)
        retrieval_observation.update(output=_sources_payload(sources))

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
    ) as generation_observation:
        response = _generate_answer(question, sources, settings)
        generation_observation.update(output=str(response.content))

    return Answer(content=str(response.content), sources=sources)


def _search_sources(question: str, settings: Settings) -> list[Source]:
    vector_store = _vector_store(settings)
    search_results = vector_store.similarity_search_with_relevance_scores(question, k=4)

    min_score = 0.30
    filtered_results = [
        (document, score)
        for document, score in search_results
        if score >= min_score
    ]

    sources = [
        Source(content=document.page_content, heading=document.metadata["heading"], score=score)
        for document, score in filtered_results
    ]

    return sources


def _generate_answer(question: str, sources: list[Source], settings: Settings):
    context = _build_context(sources)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Ты тьютор по уроку «Работа с фреймворками и агентами».
                
                Правила ответа:
                1. Отвечай только на русском языке.
                2. Используй только информацию из контекста.
                3. Не добавляй факты, выводы, оценки и объяснения, которых нет в контексте.
                4. Не используй слова вроде «масштабируемый», «эффективный», «удобный», если этого нет в контексте.
                5. Формулируй ответ максимально близко к тексту заметок.
                6. Если информации недостаточно, ответь строго:
                «В материалах урока нет информации для ответа на этот вопрос.»
                7. Не упоминай источники внутри ответа — они показываются отдельно.
                """.strip(),
                            ),
                            (
                                "human",
                                """
                Контекст:
                {context}
                
                Вопрос:
                {question}
                
                Ответь кратко и строго по контексту.
                """.strip(),
            ),
        ]
    )

    model = ChatOllama(
        model=settings.chat_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )

    response = (prompt | model).invoke(
        {
            "context": context,
            "question": question,
        }
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
    source_text = settings.source_file.read_text(encoding="utf-8")
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section"), ("###", "subsection")],
        strip_headers=False,
    )
    sections = header_splitter.split_text(source_text)
    chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
    )
    chunks = chunk_splitter.split_documents(sections)

    for chunk_index, chunk in enumerate(chunks):
        heading = chunk.metadata.get("subsection") or chunk.metadata.get("section") or chunk.metadata.get("title") or "Без заголовка"
        chunk_id = sha256(f"{chunk_index}:{chunk.page_content}".encode("utf-8")).hexdigest()[:12]
        chunk.metadata = {"chunk_id": chunk_id, "heading": heading}

    return chunks


def _embeddings(settings: Settings) -> OllamaEmbeddings:
    return OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_base_url)


def _vector_store(settings: Settings) -> Chroma:
    return Chroma(
        collection_name="lesson_12",
        embedding_function=_embeddings(settings),
        persist_directory=str(settings.chroma_directory),
    )

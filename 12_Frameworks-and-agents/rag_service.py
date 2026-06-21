from dataclasses import dataclass
from hashlib import sha256
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

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

    vector_store = _vector_store(settings)
    search_results = vector_store.similarity_search_with_relevance_scores(question, k=4)
    sources = [
        Source(content=document.page_content, heading=document.metadata["heading"], score=score)
        for document, score in search_results
    ]
    context = "\n\n".join(f"[{source.heading}]\n{source.content}" for source in sources)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Ты тьютор по уроку «Работа с фреймворками и агентами». "
                "Отвечай только на русском и только по контексту. "
                "Если в контексте нет ответа, скажи: «В материалах урока нет информации для ответа на этот вопрос.»",
            ),
            ("human", "Контекст:\n{context}\n\nВопрос: {question}"),
        ]
    )
    model = ChatOllama(model=settings.chat_model, base_url=settings.ollama_base_url, temperature=0)
    response = (prompt | model).invoke({"context": context, "question": question})

    return Answer(content=str(response.content), sources=sources)


def _load_chunks(settings: Settings) -> list[Document]:
    source_text = settings.source_file.read_text(encoding="utf-8")
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section"), ("###", "subsection")],
        strip_headers=False,
    )
    sections = header_splitter.split_text(source_text)
    chunk_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
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

from dataclasses import dataclass
from pathlib import Path
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import Settings


@dataclass
class SourceFragment:
    title: str
    relevance: float
    text: str
    metadata: dict


@dataclass
class RagAnswer:
    answer: str
    sources: list[SourceFragment]


class RagService:
    def __init__(self) -> None:
        self._embeddings = OllamaEmbeddings(
            model=Settings.OLLAMA_EMBEDDING_MODEL,
            base_url=Settings.OLLAMA_BASE_URL,
        )

        self._chat_model = ChatOllama(
            model=Settings.OLLAMA_CHAT_MODEL,
            base_url=Settings.OLLAMA_BASE_URL,
            temperature=0,
        )

    def create_index(
        self,
        markdown_paths: list[Path],
    ) -> int:
        Settings.ensure_directories()

        if Settings.CHROMA_DIR.exists():
            shutil.rmtree(Settings.CHROMA_DIR)

        Settings.CHROMA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        documents = self._load_markdown_documents(
            markdown_paths=markdown_paths,
        )

        if not documents:

            return 0

        Chroma.from_documents(
            documents=documents,
            embedding=self._embeddings,
            persist_directory=str(Settings.CHROMA_DIR),
        )

        return len(documents)

    def answer_question(
        self,
        question: str,
        top_k: int = 4,
    ) -> RagAnswer:
        vector_store = self._load_vector_store()

        documents_with_scores = vector_store.similarity_search_with_relevance_scores(
            query=question,
            k=top_k,
        )

        sources = [
            SourceFragment(
                title=self._get_source_title(document),
                relevance=float(score),
                text=document.page_content,
                metadata=document.metadata,
            )
            for document, score in documents_with_scores
        ]

        if not sources:

            return RagAnswer(
                answer="В загруженных документах нет информации для ответа на этот вопрос.",
                sources=[],
            )

        context = self._build_context(
            sources=sources,
        )

        prompt = self._build_prompt(
            question=question,
            context=context,
        )

        response = self._chat_model.invoke(prompt)
        answer = str(response.content).strip()

        if not answer:
            answer = "В загруженных документах нет информации для ответа на этот вопрос."

        return RagAnswer(
            answer=answer,
            sources=sources,
        )

    def index_exists(self) -> bool:
        if not Settings.CHROMA_DIR.exists():

            return False

        sqlite_path = Settings.CHROMA_DIR / "chroma.sqlite3"

        return sqlite_path.exists()

    def _load_vector_store(self) -> Chroma:
        vector_store = Chroma(
            persist_directory=str(Settings.CHROMA_DIR),
            embedding_function=self._embeddings,
        )

        return vector_store

    def _load_markdown_documents(
        self,
        markdown_paths: list[Path],
    ) -> list[Document]:
        documents: list[Document] = []

        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "document_title"),
                ("##", "section_title"),
            ],
            strip_headers=False,
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
        )

        for markdown_path in markdown_paths:
            markdown_text = markdown_path.read_text(
                encoding="utf-8",
            )

            header_documents = header_splitter.split_text(markdown_text)
            split_documents = text_splitter.split_documents(header_documents)

            for chunk_index, document in enumerate(split_documents):
                document.metadata["source_file"] = markdown_path.name
                document.metadata["chunk_id"] = (
                    f"{markdown_path.stem}-{chunk_index:04d}"
                )

                documents.append(document)

        return documents

    def _build_context(
        self,
        sources: list[SourceFragment],
    ) -> str:
        context_parts: list[str] = []

        for source in sources:
            context_parts.append(
                f"Источник: {source.title}\n{source.text}"
            )

        context = "\n\n---\n\n".join(context_parts)

        return context

    def _build_prompt(
        self,
        question: str,
        context: str,
    ) -> str:
        prompt = f"""
Ты отвечаешь на русском языке только на основании контекста из загруженных документов.

Если в контексте нет ответа, верни ровно эту фразу:
В загруженных документах нет информации для ответа на этот вопрос.

Контекст:
{context}

Вопрос:
{question}

Ответ:
""".strip()

        return prompt

    def _get_source_title(
        self,
        document: Document,
    ) -> str:
        source_file = document.metadata.get(
            "source_file",
            "unknown.md",
        )

        section_title = document.metadata.get(
            "section_title",
            "",
        )

        if section_title:
            title = f"{source_file} — {section_title}"
        else:
            title = source_file

        return title

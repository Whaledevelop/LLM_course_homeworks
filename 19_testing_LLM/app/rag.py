import hashlib
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings


class RagApplication:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._embeddings = OpenAIEmbeddings(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
        )
        self._llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.llm_model,
            temperature=0,
        )
        self._vector_store = self._create_vector_store()

    def ask(self, question: str) -> dict[str, Any]:
        documents = self._vector_store.similarity_search(
            question,
            k=self._settings.retrieved_chunks,
        )
        contexts = [document.page_content for document in documents]
        context = "\n\n---\n\n".join(contexts)
        response = self._llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Ответь на вопрос только по предоставленному контексту. "
                        "Если в контексте нет ответа, прямо сообщи об этом. "
                        "Отвечай кратко и технически точно на русском языке."
                    )
                ),
                HumanMessage(
                    content=f"Контекст:\n{context}\n\nВопрос:\n{question}"
                ),
            ]
        )

        return {
            "question": question,
            "answer": self._response_text(response.content),
            "contexts": contexts,
        }

    def _create_vector_store(self) -> Chroma:
        documents, corpus_hash = self._load_documents()
        collection_name = f"unity_rag_{corpus_hash[:16]}"
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(self._settings.chroma_dir),
        )
        stored_documents = vector_store.get(include=[])
        if not stored_documents["ids"]:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._settings.chunk_size,
                chunk_overlap=self._settings.chunk_overlap,
                separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
            )
            chunks = splitter.split_documents(documents)
            vector_store.add_documents(chunks)

        return vector_store

    def _load_documents(self) -> tuple[list[Document], str]:
        document_paths = sorted(self._settings.documents_dir.glob("*.md"))
        if not document_paths:
            raise FileNotFoundError(
                f"В каталоге {self._settings.documents_dir} нет Markdown-документов"
            )

        documents = []
        digest = hashlib.sha256()
        digest.update(str(self._settings.chunk_size).encode())
        digest.update(str(self._settings.chunk_overlap).encode())
        for document_path in document_paths:
            content = document_path.read_text(encoding="utf-8")
            digest.update(document_path.name.encode())
            digest.update(content.encode())
            documents.append(
                Document(
                    page_content=content,
                    metadata={"source": document_path.name},
                )
            )

        return documents, digest.hexdigest()

    @staticmethod
    def _response_text(content: Any) -> str:
        if isinstance(content, str):
            return content

        return str(content)


_application: RagApplication | None = None


def ask(question: str) -> dict[str, Any]:
    global _application

    if _application is None:
        _application = RagApplication(Settings.from_env())

    return _application.ask(question)


def main() -> None:
    question = input("Введите вопрос: ").strip()
    result = ask(question)
    print(f"\nОтвет:\n{result['answer']}")
    print("\nНайденные contexts:")
    for index, context in enumerate(result["contexts"], start=1):
        print(f"\n[{index}]\n{context}")


if __name__ == "__main__":
    main()

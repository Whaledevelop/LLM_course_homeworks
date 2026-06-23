import streamlit as st

from document_service import DocumentService
from rag_service import RagAnswer
from rag_service import RagService
from settings import Settings


def initialize_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "conversion_results" not in st.session_state:
        st.session_state.conversion_results = []


def render_sidebar(
    document_service: DocumentService,
    rag_service: RagService,
) -> None:
    st.sidebar.header("Настройки")

    st.sidebar.text_input(
        "Ollama base URL",
        value=Settings.OLLAMA_BASE_URL,
        disabled=True,
    )

    st.sidebar.text_input(
        "Chat model",
        value=Settings.OLLAMA_CHAT_MODEL,
        disabled=True,
    )

    st.sidebar.text_input(
        "Embedding model",
        value=Settings.OLLAMA_EMBEDDING_MODEL,
        disabled=True,
    )

    if rag_service.index_exists():
        st.sidebar.success("Индекс создан")
    else:
        st.sidebar.warning("Индекс не создан")

    uploaded_files = st.sidebar.file_uploader(
        "Загрузить PDF",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.sidebar.button("Сохранить и обновить индекс"):
        handle_save_and_index(
            uploaded_files=uploaded_files,
            document_service=document_service,
            rag_service=rag_service,
        )

    if st.sidebar.button("Очистить документы"):
        handle_clear_documents(
            document_service=document_service,
        )

    render_uploaded_documents(
        document_service=document_service,
    )

    render_markdown_previews(
        document_service=document_service,
    )


def handle_save_and_index(
    uploaded_files: list,
    document_service: DocumentService,
    rag_service: RagService,
) -> None:
    if not uploaded_files:
        st.sidebar.error("Сначала загрузите хотя бы один PDF.")

        return

    try:
        with st.spinner("Конвертация PDF в Markdown..."):
            conversion_results = document_service.save_and_convert_uploaded_pdfs(
                uploaded_files=uploaded_files,
            )

            st.session_state.conversion_results = conversion_results

        markdown_paths = document_service.get_markdown_paths()

        with st.spinner("Создание Chroma-индекса..."):
            chunks_count = rag_service.create_index(
                markdown_paths=markdown_paths,
            )

        st.sidebar.success(
            f"Индекс обновлён. Чанков: {chunks_count}"
        )
    except Exception as exception:
        st.sidebar.error(str(exception))


def handle_clear_documents(
    document_service: DocumentService,
) -> None:
    try:
        document_service.clear_documents_and_index()
        st.session_state.conversion_results = []
        st.session_state.messages = []
        st.sidebar.success("Документы и индекс очищены.")
    except Exception as exception:
        st.sidebar.error(str(exception))


def render_uploaded_documents(
    document_service: DocumentService,
) -> None:
    st.sidebar.header("Загруженные PDF")

    pdf_paths = document_service.get_uploaded_pdf_paths()

    if not pdf_paths:
        st.sidebar.caption("PDF пока не загружены.")

        return

    conversion_stats = {
        result.pdf_path.name: result
        for result in st.session_state.conversion_results
    }

    for pdf_path in pdf_paths:
        st.sidebar.markdown(f"**{pdf_path.name}**")

        conversion_result = conversion_stats.get(pdf_path.name)

        if conversion_result:
            st.sidebar.caption(
                f"Страниц: {conversion_result.pages_count}, изображений: {conversion_result.images_count}"
            )


def render_markdown_previews(
    document_service: DocumentService,
) -> None:
    st.sidebar.header("Markdown preview")

    markdown_paths = document_service.get_markdown_paths()

    if not markdown_paths:
        st.sidebar.caption("Markdown пока не создан.")

        return

    for markdown_path in markdown_paths:
        with st.sidebar.expander(markdown_path.name):
            markdown_text = document_service.read_markdown(
                markdown_path=markdown_path,
            )

            st.code(
                markdown_text,
                language="markdown",
            )


def render_chat(
    rag_service: RagService,
) -> None:
    st.title("PDF RAG: PDF → Markdown")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                render_sources(
                    answer=message.get("answer"),
                )

    question = st.chat_input("Задайте вопрос по загруженным PDF")

    if not question:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.spinner("Поиск ответа..."):
            rag_answer = rag_service.answer_question(
                question=question,
            )

        assistant_message = {
            "role": "assistant",
            "content": rag_answer.answer,
            "answer": rag_answer,
        }

        st.session_state.messages.append(assistant_message)

        with st.chat_message("assistant"):
            st.markdown(rag_answer.answer)
            render_sources(
                answer=rag_answer,
            )
    except Exception as exception:
        st.error(str(exception))


def render_sources(
    answer: RagAnswer | None,
) -> None:
    if not answer:
        return

    if not answer.sources:
        return

    st.markdown("#### Источники")

    for source_index, source in enumerate(answer.sources, start=1):
        with st.expander(
            f"{source_index}. {source.title} | релевантность: {source.relevance:.3f}"
        ):
            st.markdown(source.text)


def main() -> None:
    Settings.ensure_directories()
    initialize_state()

    document_service = DocumentService()
    rag_service = RagService()

    render_sidebar(
        document_service=document_service,
        rag_service=rag_service,
    )

    render_chat(
        rag_service=rag_service,
    )


if __name__ == "__main__":
    main()

import streamlit as st

from document_service import DocumentService
from rag_service import RagService
from settings import Settings


def initialize_state() -> None:
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
        st.error("Сначала загрузите хотя бы один PDF.")

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

        st.success(
            f"Индекс обновлён. Чанков: {chunks_count}"
        )
    except Exception as exception:
        st.error(str(exception))


def handle_clear_documents(
    document_service: DocumentService,
) -> None:
    try:
        document_service.clear_documents_and_index()
        st.session_state.conversion_results = []
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


def render_pdf_upload(
    document_service: DocumentService,
    rag_service: RagService,
) -> None:
    st.title("PDF RAG: PDF → Markdown")
    st.subheader("Загрузка PDF")
    st.caption("Результаты конвертации сохраняются в:")
    st.code(str(Settings.DOCUMENTS_DIR))
    st.caption("Markdown: `<имя_файла>.md`; изображения: `<имя_файла>_assets/images`.")

    uploaded_files = st.file_uploader(
        "Выберите PDF-файлы",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Сохранить и обновить индекс"):
        handle_save_and_index(
            uploaded_files=uploaded_files,
            document_service=document_service,
            rag_service=rag_service,
        )


def main() -> None:
    Settings.ensure_directories()
    initialize_state()

    document_service = DocumentService()
    rag_service = RagService()

    render_sidebar(
        document_service=document_service,
        rag_service=rag_service,
    )

    render_pdf_upload(
        document_service=document_service,
        rag_service=rag_service,
    )


if __name__ == "__main__":
    main()

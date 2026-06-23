from pathlib import Path

import streamlit as st

from document_service import DocumentService
from settings import Settings


def initialize_state() -> None:
    if "conversion_results" not in st.session_state:
        st.session_state.conversion_results = []


def render_sidebar() -> None:
    st.sidebar.header("Настройки модели")
    st.sidebar.text_input(
        "Ollama base URL",
        value=Settings.OLLAMA_BASE_URL,
        disabled=True,
    )
    st.sidebar.text_input(
        "Модель конспектирования",
        value=Settings.OLLAMA_CHAT_MODEL,
        disabled=True,
    )


def render_results() -> None:
    if not st.session_state.conversion_results:
        return

    st.subheader("Результаты")

    for result in st.session_state.conversion_results:
        st.success(
            f"{result.pdf_path.name}: {result.pages_count} стр., сохранено изображений: {result.images_count}"
        )
        st.code(str(result.markdown_path))


def render_content(
    document_service: DocumentService,
) -> None:
    st.title("PDF Summarizer")
    st.subheader("Загрузка и конспектирование PDF")

    include_metadata = st.toggle(
        "Meta Files",
        value=False,
        help="Выключено: служебные страницы и страницы только с заголовком не попадают в Markdown.",
    )
    include_images = st.toggle(
        "Import Images",
        value=True,
        help="Выключено: в Markdown сохраняется только текст без извлечения изображений.",
    )
    output_path = st.text_input(
        "Папка для результата Markdown и изображений",
        value=str(Settings.DEFAULT_OUTPUT_DIR),
    )
    st.caption("Для каждого PDF будут сохранены `<имя>.md`, исходный PDF и папка `<имя>_assets/images`.")

    uploaded_files = st.file_uploader(
        "Выберите PDF-файлы",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Создать Markdown", type="primary"):
        convert_files(
            uploaded_files=uploaded_files,
            output_path=output_path,
            include_metadata=include_metadata,
            include_images=include_images,
            document_service=document_service,
        )

    render_results()


def convert_files(
    uploaded_files: list,
    output_path: str,
    include_metadata: bool,
    include_images: bool,
    document_service: DocumentService,
) -> None:
    if not uploaded_files:
        st.error("Сначала загрузите хотя бы один PDF.")

        return

    if not output_path.strip():
        st.error("Укажите папку для сохранения результата.")

        return

    try:
        with st.spinner("Извлечение данных и создание Markdown..."):
            results = document_service.convert_uploaded_pdfs(
                uploaded_files=uploaded_files,
                output_dir=Path(output_path),
                include_metadata=include_metadata,
                include_images=include_images,
            )

        st.session_state.conversion_results = results
    except Exception as exception:
        st.error(str(exception))


def main() -> None:
    initialize_state()
    render_sidebar()

    document_service = DocumentService()
    render_content(document_service)


if __name__ == "__main__":
    main()

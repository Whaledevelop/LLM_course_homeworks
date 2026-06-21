from uuid import uuid4

import streamlit as st

from document_service import clear_documents, get_documents_context, get_source_files, get_source_name, save_uploaded_files
from evaluation_service import create_evaluation_dataset, run_evaluation
from observability import get_langfuse_client
from rag_service import answer_question, create_index, index_exists
from settings import get_settings


def show_sources(sources: list) -> None:
    with st.expander(f"Источники ({len(sources)})"):
        for source in sources:
            st.markdown(f"**{source.heading}** · релевантность: `{source.score:.3f}`")
            st.write(source.content)


def show_documents_panel() -> None:
    st.subheader("Документы")
    upload_version = st.session_state.get("upload_version", 0)
    uploaded_files = st.file_uploader(
        "Добавить файлы",
        type=["md", "txt"],
        accept_multiple_files=True,
        key=f"file_upload_{upload_version}",
    )
    uploaded_directory = st.file_uploader(
        "Добавить папку",
        type=["md", "txt"],
        accept_multiple_files="directory",
        key=f"directory_upload_{upload_version}",
    )
    if st.button("Сохранить и обновить индекс", use_container_width=True):
        files_to_save = uploaded_files + uploaded_directory
        saved_count = save_uploaded_files(files_to_save, settings)
        source_files = get_source_files(settings)
        if not source_files:
            st.error("Выберите хотя бы один файл .md или .txt.")
        else:
            with st.spinner("Обновляю индекс документов..."):
                chunk_count = create_index(settings)

            if saved_count > 0:
                st.success(f"Добавлено файлов: {saved_count}. Индекс: {chunk_count} фрагментов.")
            else:
                st.success(f"Индекс обновлён: {chunk_count} фрагментов.")

    if st.button("Clear files", use_container_width=True):
        clear_documents(settings)
        st.session_state.upload_version = upload_version + 1
        st.session_state.files_cleared = True
        st.rerun()

    if st.session_state.pop("files_cleared", False):
        st.success("Документы и индекс очищены.")

    source_files = get_source_files(settings)
    st.caption(f"В контексте файлов: {len(source_files)}")
    for source_file in source_files:
        st.write(f"• {get_source_name(source_file, settings)}")

    st.subheader("Полный контекст")
    with st.container(height=420):
        st.code(get_documents_context(settings), language="markdown")


settings = get_settings()
st.set_page_config(page_title="Справочник по документам", page_icon="📚")
st.title("Справочник по загруженным документам")
st.caption("Локальный RAG: Ollama + LangChain + Chroma")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Настройки")
    st.write(f"Чат-модель: `{settings.chat_model}`")
    st.write(f"Embedding-модель: `{settings.embedding_model}`")
    st.write(f"Индекс: {'готов' if index_exists(settings) else 'не создан'}")

    if get_langfuse_client(settings) is None:
        st.caption("Langfuse выключен: добавьте ключи в .env")
    else:
        if st.button("Создать датасет Langfuse", use_container_width=True):
            try:
                created = create_evaluation_dataset(settings)
                if created:
                    message = "Датасет Langfuse создан."
                else:
                    message = "Датасет Langfuse уже существует."

                st.success(message)
            except Exception as error:
                st.error(str(error))

        if st.button("Запустить LLM-оценку", use_container_width=True):
            try:
                evaluation_results = run_evaluation(settings)
                for evaluation_result in evaluation_results:
                    st.write(
                        f"{evaluation_result['score']}/1 — {evaluation_result['question']}"
                    )
            except Exception as error:
                st.error(str(error))

    st.divider()
    show_documents_panel()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            show_sources(message["sources"])

question = st.chat_input("Задайте вопрос по загруженным документам")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Ищу фрагменты урока и формирую ответ..."):
                answer = answer_question(question, settings)
            st.markdown(answer.content)
            show_sources(answer.sources)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer.content, "sources": answer.sources}
            )
        except Exception as error:
            st.error(str(error))

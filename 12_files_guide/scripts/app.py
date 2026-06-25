from uuid import uuid4

import streamlit as st

from document_service import clear_documents, get_documents_context, get_source_files, get_source_name, save_uploaded_files
from evaluation_service import create_evaluation_dataset, run_evaluation
from observability import create_event, get_langfuse_client
from rag_service import answer_question, create_index, index_exists
from settings import get_settings


def show_sources(sources: list) -> None:
    with st.expander(f"Источники ({len(sources)})"):
        for source in sources:
            st.markdown(f"**{source.heading}** · релевантность: `{source.score:.3f}`")
            st.write(source.content)


def show_error(error: Exception, operation: str) -> None:
    create_event(
        settings,
        name="error",
        input_data={"operation": operation},
        output_data={"error": str(error)},
        level="ERROR",
        status_message=str(error),
    )
    st.error(str(error))


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
        try:
            files_to_save = uploaded_files + uploaded_directory
            saved_count = save_uploaded_files(files_to_save, settings)
            source_files = get_source_files(settings)
            if not source_files:
                raise RuntimeError("Выберите хотя бы один файл .md или .txt.")

            if saved_count > 0:
                create_event(
                    settings,
                    name="documents_uploaded",
                    input_data={"selected_file_count": len(files_to_save)},
                    output_data={"saved_file_count": saved_count},
                )

            with st.spinner("Обновляю индекс документов..."):
                chunk_count = create_index(settings)

            create_event(
                settings,
                name="index_created",
                input_data={"document_count": len(source_files)},
                output_data={"chunk_count": chunk_count},
            )
            if saved_count > 0:
                st.success(f"Добавлено файлов: {saved_count}. Индекс: {chunk_count} фрагментов.")
            else:
                st.success(f"Индекс обновлён: {chunk_count} фрагментов.")
        except Exception as error:
            show_error(error, "save_and_index")

    if st.button("Clear files", use_container_width=True):
        try:
            cleared_file_count = len(get_source_files(settings))
            clear_documents(settings)
            create_event(
                settings,
                name="files_cleared",
                input_data={"file_count": cleared_file_count},
            )
            st.session_state.upload_version = upload_version + 1
            st.session_state.files_cleared = True
            st.rerun()
        except Exception as error:
            show_error(error, "clear_files")

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
st.caption("RAG: Hugging Face + LangChain + Chroma")

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
                show_error(error, "create_evaluation_dataset")

        if st.button("Запустить LLM-оценку", use_container_width=True):
            try:
                evaluation_results = run_evaluation(settings)
                for evaluation_result in evaluation_results:
                    st.write(
                        f"{evaluation_result['score']}/1 — {evaluation_result['question']}"
                    )
            except Exception as error:
                show_error(error, "run_evaluation")

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
            show_error(error, "answer_question")

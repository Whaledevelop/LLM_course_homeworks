from uuid import uuid4

import streamlit as st

from rag_service import answer_question, create_index, index_exists
from settings import get_settings


def show_sources(sources: list) -> None:
    with st.expander(f"Источники ({len(sources)})"):
        for source in sources:
            st.markdown(f"**{source.heading}** · релевантность: `{source.score:.3f}`")
            st.write(source.content)


settings = get_settings()
st.set_page_config(page_title="Тьютор по уроку 12", page_icon="📚")
st.title("Тьютор по уроку 12: фреймворки и агенты")
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

    if st.button("Создать индекс", use_container_width=True):
        try:
            with st.spinner("Читаю конспект и создаю векторы..."):
                chunk_count = create_index(settings)
            st.success(f"Индекс готов: {chunk_count} фрагментов.")
        except Exception as error:
            st.error(str(error))

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            show_sources(message["sources"])

question = st.chat_input("Задайте вопрос по уроку 12")
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

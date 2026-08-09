from langchain_core.embeddings import Embeddings
from yandex_ai_studio_sdk import AIStudio


class YandexAIStudioEmbeddings(Embeddings):
    def __init__(self, api_key: str, folder_id: str, model: str) -> None:
        sdk = AIStudio(folder_id=folder_id, auth=api_key)
        self._document_model = sdk.models.text_embeddings(f"{model}-doc")
        self._query_model = sdk.models.text_embeddings(f"{model}-query")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            result = self._document_model.run(text)
            embeddings.append(list(result.embedding))

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        result = self._query_model.run(text)

        return list(result.embedding)

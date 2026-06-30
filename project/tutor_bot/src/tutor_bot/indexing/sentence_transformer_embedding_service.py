from collections.abc import Sequence

from sentence_transformers import SentenceTransformer


_DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"


class SentenceTransformerEmbeddingService:
    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("Embedding batch size must be positive")

        self._model = SentenceTransformer(
            model_name,
            device=device,
        )
        self._batch_size = batch_size

    @property
    def dimension(self) -> int:
        dimension = self._model.get_embedding_dimension()

        if dimension is None:
            raise RuntimeError("Embedding model dimension is unavailable")

        return dimension

    def embed_passages(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        prepared_texts = [f"passage: {text}" for text in texts]

        return self._encode(prepared_texts)

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        embeddings = self._encode([f"query: {text}"])

        return embeddings[0]

    def _encode(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        embeddings = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

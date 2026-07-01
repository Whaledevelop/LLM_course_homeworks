from collections.abc import Sequence

from sentence_transformers import CrossEncoder


_DEFAULT_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class SentenceTransformerReranker:
    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        batch_size: int = 16,
        device: str | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("Reranker batch size must be positive")

        self._model = CrossEncoder(
            model_name,
            device=device,
        )
        self._batch_size = batch_size

    def score(
        self,
        query: str,
        passages: Sequence[str],
    ) -> list[float]:
        if not passages:
            return []

        query_passage_pairs = [(query, passage) for passage in passages]

        scores = self._model.predict(
            query_passage_pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )

        return [float(score) for score in scores]

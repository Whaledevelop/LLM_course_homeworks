from collections.abc import Sequence
from typing import Protocol


class EmbeddingService(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_passages(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]: ...

    def embed_query(
        self,
        text: str,
    ) -> list[float]: ...

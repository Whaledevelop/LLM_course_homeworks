from collections.abc import Sequence
from typing import Protocol


class Reranker(Protocol):
    def score(
        self,
        query: str,
        passages: Sequence[str],
    ) -> list[float]: ...

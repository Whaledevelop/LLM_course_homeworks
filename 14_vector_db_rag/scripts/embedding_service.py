from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


class HashingEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def __call__(self, input: Documents) -> Embeddings:
        return [self.embed(document) for document in input]

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions

        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector

        return [value / norm for value in vector]

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def _tokens(self, text: str) -> list[str]:
        return re.findall(r"[a-zа-яё0-9]+", text.lower())

from functools import lru_cache

from langfuse import Langfuse

from settings import Settings


@lru_cache
def get_langfuse_client(settings: Settings) -> Langfuse | None:
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None

    client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )

    return client

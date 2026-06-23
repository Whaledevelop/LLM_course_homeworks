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


def create_event(
    settings: Settings,
    name: str,
    input_data: dict | None = None,
    output_data: dict | None = None,
    level: str = "DEFAULT",
    status_message: str | None = None,
) -> None:
    langfuse_client = get_langfuse_client(settings)
    if langfuse_client is not None:
        try:
            langfuse_client.create_event(
                name=name,
                input=input_data,
                output=output_data,
                level=level,
                status_message=status_message,
            )
            langfuse_client.flush()
        except Exception:
            pass

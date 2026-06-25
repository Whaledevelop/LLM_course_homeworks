from dataclasses import dataclass
from functools import lru_cache

from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings

from settings import Settings


@dataclass(frozen=True)
class LlmResponse:
    content: str
    usage_details: dict[str, int] | None


@lru_cache
def _client(
    token: str | None,
    provider: str,
    timeout: float,
) -> InferenceClient:
    return InferenceClient(
        provider=provider,
        token=token,
        timeout=timeout,
    )


def generate_chat_response(
    messages: list[dict],
    settings: Settings,
    max_tokens: int,
    temperature: float = 0,
) -> LlmResponse:
    _require_hf_token(settings)
    client = _client(settings.hf_token, settings.hf_provider, settings.hf_timeout)
    response = client.chat_completion(
        messages=messages,
        model=settings.chat_model,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = response.choices[0].message.content or ""

    return LlmResponse(
        content=content,
        usage_details=_usage_details(response),
    )


class HuggingFaceInferenceEmbeddings(Embeddings):
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = _client(settings.hf_token, settings.hf_provider, settings.hf_timeout)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        _require_hf_token(self._settings)
        embeddings = self._client.feature_extraction(
            texts,
            model=self._settings.embedding_model,
            normalize=True,
            truncate=True,
        )

        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        _require_hf_token(self._settings)
        embeddings = self._client.feature_extraction(
            text,
            model=self._settings.embedding_model,
            normalize=True,
            truncate=True,
        )

        embedding = embeddings.tolist()
        if embedding and isinstance(embedding[0], list):
            return embedding[0]

        return embedding


def _require_hf_token(settings: Settings) -> None:
    if not settings.hf_token:
        raise RuntimeError("Добавьте HF_TOKEN в .env для доступа к Hugging Face Inference Providers.")


def _usage_details(response) -> dict[str, int] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    input_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    if input_tokens is None or output_tokens is None:
        return None

    usage_details = {
        "input": int(input_tokens),
        "output": int(output_tokens),
        "total": int(total_tokens or input_tokens + output_tokens),
    }

    return usage_details

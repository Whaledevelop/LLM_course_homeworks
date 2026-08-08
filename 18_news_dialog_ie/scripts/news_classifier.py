import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT_VERSION = "local-news-classifier-v1"
INSTRUCTION = """Classify the dialog as NEWS or NOT_NEWS.

NEWS:
The dialog mainly discusses, summarizes or asks about a real or supposed news event, news article, or current public/political/economic development.

NOT_NEWS:
fiction, chapter/story review, programming, roleplay, creative writing, marketing, generic educational text, random conversation, or incidental mentions of news-related words.

Reply with exactly one token:
NEWS
or
NOT_NEWS"""
VALID_CLASSIFICATIONS = {"NEWS", "NOT_NEWS"}


class NewsClassifier:
    def __init__(
        self,
        cache_path: Path,
        model: str = "",
        batch_size: int = 8,
        loader: Callable | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("News classifier batch size must be positive.")
        self.model = model or os.getenv("NEWS_CLASSIFIER_MODEL", "") or DEFAULT_MODEL
        self.batch_size = batch_size
        self.device = detect_device()
        self.load_seconds = 0.0
        self._cache_path = cache_path
        self._cache = self._load_cache()
        self._loader = loader or load_local_generator
        self._generator = None

    def classify(self, dialog_id: str, text: str) -> tuple[str, bool]:
        return self.classify_batch([(dialog_id, text)])[0]

    def classify_batch(self, dialogs: list[tuple[str, str]]) -> list[tuple[str, bool]]:
        results: list[tuple[str, bool] | None] = [None] * len(dialogs)
        pending = []
        for index, (dialog_id, text) in enumerate(dialogs):
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            cache_key = self._cache_key(text_hash)
            cached = self._cache.get(cache_key)
            if cached:
                results[index] = (cached["classification"], True)
            else:
                pending.append((index, dialog_id, text, text_hash, cache_key))
        if pending:
            self._ensure_loaded()
            responses = self._generator([build_prompt(item[2]) for item in pending])
            if len(responses) != len(pending):
                raise RuntimeError("Local news classifier returned an unexpected number of responses.")
            records = []
            for pending_item, response in zip(pending, responses):
                index, dialog_id, _, text_hash, cache_key = pending_item
                classification = parse_classification(response)
                record = {
                    "dialog_id": dialog_id,
                    "text_hash": text_hash,
                    "model": self.model,
                    "prompt_version": PROMPT_VERSION,
                    "classification": classification,
                    "classified_at": datetime.now(timezone.utc).isoformat(),
                }
                records.append(record)
                self._cache[cache_key] = record
                results[index] = (classification, False)
            self._append_cache(records)

        return [result for result in results if result is not None]

    def _ensure_loaded(self) -> None:
        if self._generator is not None:
            return
        self._generator, self.device, self.load_seconds = self._loader(self.model)

    def _cache_key(self, text_hash: str) -> str:
        return f"{text_hash}:{self.model}:{PROMPT_VERSION}"

    def _load_cache(self) -> dict[str, dict]:
        if not self._cache_path.exists():
            return {}
        cache = {}
        with self._cache_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    record = json.loads(line)
                    cache_key = f"{record['text_hash']}:{record['model']}:{record.get('prompt_version', PROMPT_VERSION)}"
                    cache[cache_key] = record
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    raise RuntimeError(f"Invalid news classifier cache at line {line_number}.") from error

        return cache

    def _append_cache(self, records: list[dict]) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_path.open("a", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_local_generator(model_id: str) -> tuple[Callable, str, float]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started_at = time.perf_counter()
    device = select_device(torch.cuda.is_available())
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()

    def generate(prompts: list[str]) -> list[str]:
        formatted_prompts = [format_prompt(tokenizer, prompt) for prompt in prompts]
        encoded = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=False)
        encoded = {name: value.to(device) for name, value in encoded.items()}
        input_length = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=4,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_ids = output_ids[:, input_length:]

        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return generate, device, time.perf_counter() - started_at


def format_prompt(tokenizer, prompt: str) -> str:
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    return prompt


def build_prompt(text: str) -> str:
    return f"{INSTRUCTION}\n\nDialog:\n{text}"


def parse_classification(value: str) -> str:
    normalized = value.strip()

    return normalized if normalized in VALID_CLASSIFICATIONS else "INVALID"


def select_device(cuda_available: bool) -> str:
    return "cuda" if cuda_available else "cpu"


def detect_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    return select_device(torch.cuda.is_available())

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
PROMPT_VERSION = "local-news-classifier-v2"
INSTRUCTION = """Classify the dialog as NEWS or NOT_NEWS.

NEWS:
The dialog discusses a real-world news event or contains a real news article. This includes summaries, analysis, or fact-checking of real news reports; current political, economic, social, or public events; and requests such as "what major news happened" or "latest developments".

NOT_NEWS:
Fiction, hypothetical scenarios, alternate history, creative writing, novel or chapter reviews, programming, troubleshooting, games, roleplay, fake or fictional articles, academic or literature reviews, generic historical or educational questions, product troubleshooting, and incidental use of words such as reported, date, war, event, or article.

An article is NEWS only if it describes a real-world news event. A fictional, fake, hypothetical, academic or historical article is NOT_NEWS.

Reply with exactly one label: NEWS or NOT_NEWS."""
VALID_CLASSIFICATIONS = {"NEWS", "NOT_NEWS"}
SANITY_EXAMPLES = (
    ("Reuters reported that Apple announced layoffs on Monday.", "NEWS"),
    ("What major news happened on January 23, 2023?", "NEWS"),
    ("Summarize this BBC report about the earthquake in Turkey.", "NEWS"),
    ("Fact-check this report about a court ruling involving Donald Trump.", "NEWS"),
    ("What are the latest developments in the war in Ukraine?", "NEWS"),
    ("AP reports that parliament approved the new budget today.", "NEWS"),
    ("Write a fantasy story about a king.", "NOT_NEWS"),
    ("My Apple Magic Mouse disconnects on Debian.", "NOT_NEWS"),
    ("Critically review Chapter 4 of this novel.", "NOT_NEWS"),
    ("Write a fake news article about Batman.", "NOT_NEWS"),
    ("Explain how a combustion engine works.", "NOT_NEWS"),
    ("Design a boss fight for a video game.", "NOT_NEWS"),
)


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

    def classify(self, dialog_id: str, text: str) -> tuple[str, bool, str]:
        return self.classify_batch([(dialog_id, text)])[0]

    def classify_batch(self, dialogs: list[tuple[str, str]]) -> list[tuple[str, bool, str]]:
        results: list[tuple[str, bool, str] | None] = [None] * len(dialogs)
        pending = []
        for index, (dialog_id, text) in enumerate(dialogs):
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            cache_key = self._cache_key(text_hash)
            cached = self._cache.get(cache_key)
            if cached:
                results[index] = (cached["classification"], True, cached.get("raw_output", ""))
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
                    "raw_output": response,
                    "classified_at": datetime.now(timezone.utc).isoformat(),
                }
                records.append(record)
                self._cache[cache_key] = record
                results[index] = (classification, False, response)
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
    load_kwargs = build_model_load_kwargs(torch, device)
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    input_device = resolve_model_input_device(model)
    print_model_diagnostics(model, torch, input_device)
    load_seconds = time.perf_counter() - started_at
    print(f"[Classifier] Load time: {load_seconds:.2f} seconds")

    def generate(prompts: list[str]) -> list[str]:
        formatted_prompts = [format_prompt(tokenizer, prompt) for prompt in prompts]
        encoded = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=False)
        encoded = {name: value.to(input_device) for name, value in encoded.items()}
        input_length = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=8,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_ids = output_ids[:, input_length:]

        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return generate, str(input_device), load_seconds


def build_model_load_kwargs(torch_module, device: str) -> dict:
    load_kwargs = {
        "dtype": torch_module.float16 if device == "cuda" else torch_module.float32,
        "low_cpu_mem_usage": True,
    }
    if device == "cuda":
        load_kwargs["device_map"] = "auto"

    return load_kwargs


def resolve_model_input_device(model):
    embeddings = model.get_input_embeddings()
    if embeddings is not None and hasattr(embeddings, "weight"):
        embedding_device = embeddings.weight.device
        if str(embedding_device) != "meta":
            return embedding_device
    for parameter in model.parameters():
        if str(parameter.device) != "meta":
            return parameter.device

    raise RuntimeError("Could not determine the news classifier input device.")


def print_model_diagnostics(model, torch_module, input_device) -> None:
    device_map = getattr(model, "hf_device_map", None)
    print(f"[Classifier] Loaded input device: {input_device}")
    print(f"[Classifier] Device map: {device_map if device_map is not None else 'single-device'}")
    if str(input_device).startswith("cuda"):
        allocated_mb = torch_module.cuda.memory_allocated(input_device) / (1024 * 1024)
        reserved_mb = torch_module.cuda.memory_reserved(input_device) / (1024 * 1024)
        print(f"[Classifier] VRAM allocated: {allocated_mb:.1f} MB")
        print(f"[Classifier] VRAM reserved: {reserved_mb:.1f} MB")


def format_prompt(tokenizer, prompt: str) -> str:
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    return prompt


def build_prompt(text: str) -> str:
    return f"{INSTRUCTION}\n\nDialog:\n{text}"


def parse_classification(value: str) -> str:
    normalized = value.strip()
    code_block = re.fullmatch(r"```(?:\w+)?\s*(.*?)\s*```", normalized, re.DOTALL)
    if code_block:
        normalized = code_block.group(1).strip()
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if not lines:
        return "INVALID"
    normalized = lines[0].strip("`'\"*~ ").removesuffix(".").strip()

    return normalized if normalized in VALID_CLASSIFICATIONS else "INVALID"


def run_sanity_check(classifier: NewsClassifier, minimum_accuracy: float = 0.8, output=print) -> dict:
    dialogs = [(f"sanity-{index}", text) for index, (text, _) in enumerate(SANITY_EXAMPLES)]
    results = classifier.classify_batch(dialogs)
    news_total = sum(expected == "NEWS" for _, expected in SANITY_EXAMPLES)
    not_news_total = len(SANITY_EXAMPLES) - news_total
    news_correct = 0
    not_news_correct = 0
    invalid = 0
    for (_, expected), (classification, _, _) in zip(SANITY_EXAMPLES, results):
        invalid += int(classification == "INVALID")
        news_correct += int(expected == "NEWS" and classification == expected)
        not_news_correct += int(expected == "NOT_NEWS" and classification == expected)
    accuracy = (news_correct + not_news_correct) / len(SANITY_EXAMPLES)
    stats = {
        "news_correct": news_correct,
        "news_total": news_total,
        "not_news_correct": not_news_correct,
        "not_news_total": not_news_total,
        "invalid": invalid,
        "accuracy": accuracy,
    }
    output("Classifier sanity check")
    output(f"NEWS correct: {news_correct}/{news_total}")
    output(f"NOT_NEWS correct: {not_news_correct}/{not_news_total}")
    output(f"INVALID: {invalid}")
    if accuracy < minimum_accuracy:
        raise RuntimeError(
            f"News classifier sanity check failed: accuracy {accuracy:.1%} is below {minimum_accuracy:.0%}."
        )

    return stats


def select_device(cuda_available: bool) -> str:
    return "cuda" if cuda_available else "cpu"


def detect_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    return select_device(torch.cuda.is_available())

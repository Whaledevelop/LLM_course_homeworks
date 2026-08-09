import json
import re
import time
from abc import ABC, abstractmethod

from schemas import ExtractedItem, ExtractionResult, NewsDialog


ALLOWED_ENTITY_LABELS = {"PERSON", "ORG", "LOC", "DATE", "IMPACT", "SOURCE"}
PROMPT_VERSION = "news-ie-v2"


DATE_PATTERN = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
    r"Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},\s+\d{4}\b|\b\d{4}\b"
)
ORG_PATTERN = re.compile(
    r"\b(?:Reuters|AP News|BBC|CNN|The Guardian|NATO|Apple|Tesla|Microsoft|European Union|EU|"
    r"OpenAI|Google|Meta|United Nations|World Health Organization|Federal Reserve)\b"
)
PERSON_PATTERN = re.compile(r"\b(?:President\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b")
LOC_PATTERN = re.compile(
    r"\b(?:Washington|California|Texas|Taiwan|Hualien|Brussels|Ukraine|United States|China|"
    r"London|Paris|Berlin|Moscow|Kyiv|Gaza|Israel)\b"
)
EVENT_PATTERN = re.compile(
    r"\b(?:meeting|summit|announcement|earthquake|investigation|layoffs?|strike|election|"
    r"flood|attack|court ruling|share buyback|quarterly earnings)\b",
    re.IGNORECASE,
)
IMPACT_PATTERN = re.compile(
    r"\b(?:boosted investor confidence|strengthened NATO coordination|damaging buildings|"
    r"disrupting transport|affect production planning|harmed competition|lifted .*? shares|"
    r"signal cost pressure)\b",
    re.IGNORECASE,
)


class BaseExtractor(ABC):
    name: str
    model_name = ""
    precision_mode = "native"
    load_seconds = 0.0
    prompt_version = PROMPT_VERSION
    generation_config: dict = {}
    _inference_progress_callback = None

    def set_inference_progress_callback(self, callback) -> None:
        self._inference_progress_callback = callback

    def _report_inference_progress(self, message: str) -> None:
        if self._inference_progress_callback is not None:
            self._inference_progress_callback(message)

    @abstractmethod
    def extract_batch(self, dialogs: list[NewsDialog]) -> list[ExtractionResult]:
        raise NotImplementedError


class RuleBasedNewsExtractor(BaseExtractor):
    name = "rules"

    def extract_batch(self, dialogs: list[NewsDialog]) -> list[ExtractionResult]:
        results = []
        for dialog in dialogs:
            entities = []
            entities.extend(find_items(dialog.text, "DATE", DATE_PATTERN))
            entities.extend(find_items(dialog.text, "ORG", ORG_PATTERN))
            entities.extend(find_items(dialog.text, "PERSON", PERSON_PATTERN, blocked_values={"AP News", "The Guardian", "European Union"}))
            entities.extend(find_items(dialog.text, "LOC", LOC_PATTERN))
            entities.extend(find_items(dialog.text, "SOURCE", ORG_PATTERN, source_only=True))
            events = find_items(dialog.text, "EVENT", EVENT_PATTERN)
            impacts = find_items(dialog.text, "IMPACT", IMPACT_PATTERN)
            entities.extend(impacts)
            relations = build_relations(events, impacts, entities)
            results.append(ExtractionResult(dialog_id=dialog.dialog_id, extractor=self.name, entities=deduplicate(entities), events=deduplicate(events), relations=relations))

        return results


class SpacyNewsExtractor(BaseExtractor):
    name = "spacy"

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        import spacy

        self._nlp = spacy.load(model_name)
        self._rules = RuleBasedNewsExtractor()

    def extract_batch(self, dialogs: list[NewsDialog]) -> list[ExtractionResult]:
        rule_results = self._rules.extract_batch(dialogs)
        results = []
        for dialog, rule_result in zip(dialogs, rule_results):
            document = self._nlp(dialog.text)
            entities = list(rule_result.entities)
            for entity in document.ents:
                label = map_spacy_label(entity.label_)
                if label:
                    entities.append(ExtractedItem(label=label, value=entity.text, start=entity.start_char, end=entity.end_char, confidence=None))
            results.append(ExtractionResult(dialog_id=dialog.dialog_id, extractor=self.name, entities=deduplicate(entities), events=rule_result.events, relations=rule_result.relations))

        return results


class TransformersJsonExtractor(BaseExtractor):
    def __init__(self, model_name: str, precision_mode: str, revision: str = "main", max_new_tokens: int = 256) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
        import torch

        if precision_mode not in {"fp16", "int8"}:
            raise ValueError(f"Unsupported precision mode: {precision_mode}")
        if not torch.cuda.is_available():
            raise RuntimeError("Transformers LLM profiles require an NVIDIA CUDA device.")
        started_at = time.perf_counter()
        self.model_name = model_name
        self.precision_mode = precision_mode
        self.revision = revision
        self.name = f"{model_name}-{precision_mode}"
        self.generation_config = {"max_new_tokens": max_new_tokens, "do_sample": False}
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        load_kwargs = build_transformer_load_kwargs(precision_mode, revision, torch, BitsAndBytesConfig)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        except (RuntimeError, ValueError) as error:
            if not is_device_placement_error(error):
                raise
            torch.cuda.empty_cache()
            load_kwargs["device_map"] = "auto"
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        device_map = getattr(model, "hf_device_map", None) or {}
        self.parameter_count = model.num_parameters()
        self.cpu_offloaded_modules = sum(str(device) == "cpu" for device in device_map.values())
        self._pipeline = pipeline("text-generation", model=model, tokenizer=self._tokenizer)
        self._max_new_tokens = max_new_tokens
        self.load_seconds = time.perf_counter() - started_at
        print_transformer_diagnostics(model, torch, self.load_seconds)

    def extract_batch(self, dialogs: list[NewsDialog]) -> list[ExtractionResult]:
        self._report_inference_progress("Preparing prompt...")
        prompts = [self._format_prompt(dialog.text) for dialog in dialogs]
        self._report_inference_progress("Starting generation...")
        generation_started_at = time.perf_counter()
        responses = self._pipeline(
            prompts,
            batch_size=len(dialogs),
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        generation_seconds = time.perf_counter() - generation_started_at
        self._report_inference_progress(f"Generation finished in {generation_seconds:.1f} sec")
        self._report_inference_progress("Parsing response...")
        results = []
        for dialog, response in zip(dialogs, responses):
            generated_text = response[0]["generated_text"] if isinstance(response, list) else response["generated_text"]
            results.append(parse_llm_response(dialog.dialog_id, self.name, generated_text))

        return results

    def _format_prompt(self, text: str) -> str:
        instruction = build_prompt(text)
        if self._tokenizer.chat_template:
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if "qwen" in self.model_name.lower():
                template_kwargs["enable_thinking"] = False

            return self._tokenizer.apply_chat_template([{"role": "user", "content": instruction}], **template_kwargs)

        return instruction


def is_device_placement_error(error: Exception) -> bool:
    message = str(error).lower()

    return "out of memory" in message or "dispatched on the cpu or the disk" in message or "device_map" in message


def build_transformer_load_kwargs(precision_mode: str, revision: str, torch, bits_and_bytes_config) -> dict:
    load_kwargs = {
        "device_map": {"": 0},
        "revision": revision,
        "dtype": torch.float16,
        "low_cpu_mem_usage": True,
    }
    if precision_mode == "int8":
        load_kwargs["quantization_config"] = bits_and_bytes_config(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )

    return load_kwargs


def print_transformer_diagnostics(model, torch, load_seconds: float) -> None:
    device_map = getattr(model, "hf_device_map", None)
    cpu_modules = sorted(module for module, device in (device_map or {}).items() if str(device) == "cpu")
    print(f"[Extractor] Model: {getattr(model, 'name_or_path', type(model).__name__)}", flush=True)
    print(f"[Extractor] Parameters: {model.num_parameters():,}", flush=True)
    print(f"[Extractor] Device map: {device_map if device_map is not None else 'single-device'}", flush=True)
    print(f"[Extractor] VRAM allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB", flush=True)
    print(f"[Extractor] VRAM reserved: {torch.cuda.memory_reserved() / 1024 / 1024:.1f} MB", flush=True)
    print(f"[Extractor] Load time: {load_seconds:.1f} sec", flush=True)
    if cpu_modules:
        print(f"[Extractor] CPU-offloaded modules (FP32): {', '.join(cpu_modules)}", flush=True)
    else:
        print("[Extractor] CPU-offloaded modules: none", flush=True)
    if device_map:
        disk_modules = sorted(module for module, device in device_map.items() if str(device) == "disk")
        if disk_modules:
            print(f"[Extractor] Disk-offloaded modules: {', '.join(disk_modules)}", flush=True)


def find_items(text: str, label: str, pattern: re.Pattern, blocked_values: set[str] | None = None, source_only: bool = False) -> list[ExtractedItem]:
    items = []
    for match in pattern.finditer(text):
        value = match.group(0)
        if label == "PERSON":
            value = clean_person(value)
        if blocked_values and value in blocked_values:
            continue
        if source_only and value not in {"Reuters", "AP News", "BBC", "CNN", "The Guardian"}:
            continue
        items.append(ExtractedItem(label=label, value=value, start=match.start(), end=match.end(), confidence=0.75))

    return items


def clean_person(value: str) -> str:
    cleaned_value = re.sub(r"^(President|Secretary General|General|Prime Minister|Minister)\s+", "", value)

    return cleaned_value


def deduplicate(items: list[ExtractedItem]) -> list[ExtractedItem]:
    seen = set()
    unique_items = []
    for item in items:
        key = (item.label, item.value.lower())
        if key in seen:
            continue
        seen.add(key)
        unique_items.append(item)

    return unique_items


def build_relations(events: list[ExtractedItem], impacts: list[ExtractedItem], entities: list[ExtractedItem]) -> list[dict[str, str]]:
    relations = []
    source = first_value(entities, "SOURCE")
    date = first_value(entities, "DATE")
    location = first_value(entities, "LOC")
    for event in events:
        relation = {"event": event.value}
        if source:
            relation["source"] = source
        if date:
            relation["date"] = date
        if location:
            relation["location"] = location
        if impacts:
            relation["impact"] = impacts[0].value
        relations.append(relation)

    return relations


def first_value(items: list[ExtractedItem], label: str) -> str:
    for item in items:
        if item.label == label:
            return item.value

    return ""


def map_spacy_label(label: str) -> str:
    mapping = {
        "PERSON": "PERSON",
        "ORG": "ORG",
        "GPE": "LOC",
        "LOC": "LOC",
        "DATE": "DATE",
        "EVENT": "EVENT",
    }

    return mapping.get(label, "")


def build_prompt(text: str) -> str:
    schema = {"entities": [{"label": "PERSON|ORG|LOC|DATE|IMPACT|SOURCE", "value": "text"}], "events": [{"label": "EVENT", "value": "text"}], "relations": [{"event": "text", "source": "text", "date": "text", "location": "text", "impact": "text"}]}

    return f"Extract news entities and events from the dialog. Return only valid JSON matching this schema: {json.dumps(schema)}\nDialog:\n{text}\nJSON:"


def parse_llm_response(dialog_id: str, extractor_name: str, generated_text: str) -> ExtractionResult:
    payload, error = extract_json(generated_text)
    if error:
        return ExtractionResult(dialog_id=dialog_id, extractor=extractor_name, raw_response=generated_text, parse_valid=False, error=error)
    entities = []
    for item in payload["entities"]:
        label = str(item.get("label", "")).upper()
        value = str(item.get("value", "")).strip()
        if label in ALLOWED_ENTITY_LABELS and value:
            entities.append(ExtractedItem(label=label, value=value))
    events = [ExtractedItem(label="EVENT", value=str(item.get("value", "")).strip()) for item in payload["events"] if str(item.get("value", "")).strip()]
    relations = [relation for relation in payload["relations"] if isinstance(relation, dict)]

    return ExtractionResult(dialog_id=dialog_id, extractor=extractor_name, entities=deduplicate(entities), events=deduplicate(events), relations=relations, raw_response=generated_text)


def extract_json(text: str) -> tuple[dict, str]:
    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        missing_fields = {"entities", "events", "relations"} - payload.keys()
        if missing_fields:
            continue
        if not all(isinstance(payload[field], list) for field in ("entities", "events", "relations")):
            return {}, "schema fields must be arrays"

        return payload, ""

    return {}, "valid extraction JSON was not found"

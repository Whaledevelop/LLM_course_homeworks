import json
import re
from abc import ABC, abstractmethod

from schemas import ExtractedItem, ExtractionResult, NewsDialog


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
    def __init__(self, model_name: str, quantized: bool, max_new_tokens: int = 256) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch

        self.name = f"{model_name}-{'int8' if quantized else 'fp16'}"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        load_kwargs = {"device_map": "auto"}
        if quantized:
            load_kwargs["load_in_8bit"] = True
        else:
            load_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self._pipeline = pipeline("text-generation", model=model, tokenizer=self._tokenizer)
        self._max_new_tokens = max_new_tokens

    def extract_batch(self, dialogs: list[NewsDialog]) -> list[ExtractionResult]:
        prompts = [build_prompt(dialog.text) for dialog in dialogs]
        responses = self._pipeline(prompts, batch_size=len(dialogs), max_new_tokens=self._max_new_tokens, do_sample=False)
        results = []
        for dialog, response in zip(dialogs, responses):
            generated_text = response[0]["generated_text"] if isinstance(response, list) else response["generated_text"]
            results.append(parse_llm_response(dialog.dialog_id, self.name, generated_text))

        return results


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
    payload = extract_json(generated_text)
    entities = [ExtractedItem(label=str(item.get("label", "")), value=str(item.get("value", ""))) for item in payload.get("entities", []) if item.get("label") and item.get("value")]
    events = [ExtractedItem(label=str(item.get("label", "EVENT")), value=str(item.get("value", ""))) for item in payload.get("events", []) if item.get("value")]
    relations = [relation for relation in payload.get("relations", []) if isinstance(relation, dict)]

    return ExtractionResult(dialog_id=dialog_id, extractor=extractor_name, entities=deduplicate(entities), events=deduplicate(events), relations=relations, raw_response=generated_text)


def extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {"entities": [], "events": [], "relations": []}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {"entities": [], "events": [], "relations": []}

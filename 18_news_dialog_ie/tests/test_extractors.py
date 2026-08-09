import json

from extractors import RuleBasedNewsExtractor, build_prompt, build_transformer_load_kwargs, configure_decoder_tokenizer, extract_json, parse_llm_response
from schemas import NewsDialog


def test_extract_json_skips_schema_example_before_result() -> None:
    text = 'schema {"entities": []} result {"entities": [], "events": [], "relations": []}'
    payload, error = extract_json(text)

    assert not error
    assert payload["events"] == []


def test_parse_llm_response_validates_labels() -> None:
    text = '{"entities":[{"label":"ORG","value":"OpenAI"},{"label":"UNKNOWN","value":"x"}],"events":[{"value":"launch"}],"relations":[]}'
    result = parse_llm_response("1", "llm", text)

    assert result.parse_valid
    assert [(item.label, item.value) for item in result.entities] == [("ORG", "OpenAI")]
    assert result.events[0].label == "EVENT"


def test_parse_llm_response_reports_invalid_json() -> None:
    result = parse_llm_response("1", "llm", "not json")

    assert not result.parse_valid
    assert result.error


def test_parse_llm_response_normalizes_safe_entity_label_aliases() -> None:
    text = '{"entities":[{"label":"ORGANIZATION","value":"OpenAI"},{"label":"LOCATION","value":"Paris"}],"events":[],"relations":[]}'

    result = parse_llm_response("1", "llm", text)

    assert [(item.label, item.value) for item in result.entities] == [("ORG", "OpenAI"), ("LOC", "Paris")]


def test_prompt_defines_labels_alias_restrictions_and_source_semantics() -> None:
    prompt = build_prompt("CNN reported an event in Paris.")

    assert "Use only these labels: PERSON, ORG, LOC, DATE, IMPACT, SOURCE, EVENT." in prompt
    assert "Never use ORGANIZATION, LOCATION" in prompt
    assert "label it SOURCE, not ORG" in prompt
    assert "not a broad topic" in prompt
    assert "Do not infer or invent facts" in prompt


def test_prompt_schema_is_valid_json_and_keeps_relations_empty() -> None:
    prompt = build_prompt("Reuters reported an event.")
    schema_text = prompt.split("Required schema: ", maxsplit=1)[1].split("\nDialog:", maxsplit=1)[0]
    schema = json.loads(schema_text)

    assert schema == {
        "entities": [{"label": "PERSON|ORG|LOC|DATE|IMPACT|SOURCE", "value": "..."}],
        "events": [{"label": "EVENT", "value": "..."}],
        "relations": [],
    }


def test_rule_extractor_handles_empty_and_repeated_entities() -> None:
    extractor = RuleBasedNewsExtractor()
    dialogs = [
        NewsDialog("empty", "test", "There are no named entities here."),
        NewsDialog("repeat", "test", "Reuters said Reuters reported a meeting in London in 2024."),
    ]
    results = extractor.extract_batch(dialogs)

    assert results[0].entities == []
    assert sum(item.value == "Reuters" and item.label == "ORG" for item in results[1].entities) == 1


class FakeTorch:
    float16 = "float16"

    class cuda:
        @staticmethod
        def mem_get_info():
            return 3 * 1024**3, 8 * 1024**3


class FakeQuantizationConfig:
    def __init__(self, **kwargs) -> None:
        self.options = kwargs


class FakeTokenizer:
    padding_side = "right"
    pad_token_id = None
    pad_token = None
    eos_token = "<eos>"


def test_decoder_tokenizer_uses_left_padding_and_eos_for_padding() -> None:
    tokenizer = FakeTokenizer()

    configure_decoder_tokenizer(tokenizer)

    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token == "<eos>"


def test_int8_load_kwargs_enable_cpu_offload_and_low_memory_loading() -> None:
    kwargs = build_transformer_load_kwargs("int8", "main", FakeTorch, FakeQuantizationConfig)

    assert kwargs["device_map"] == {"": 0}
    assert kwargs["dtype"] == "float16"
    assert kwargs["low_cpu_mem_usage"] is True
    assert kwargs["quantization_config"].options == {
        "load_in_8bit": True,
        "llm_int8_enable_fp32_cpu_offload": True,
    }
    assert "torch_dtype" not in kwargs

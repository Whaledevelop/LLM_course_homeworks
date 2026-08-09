from extractors import RuleBasedNewsExtractor, build_mistral_int8_device_map, build_transformer_load_kwargs, extract_json, parse_llm_response
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


class FakeMistralConfig:
    hidden_size = 4096
    intermediate_size = 14336
    vocab_size = 32000
    num_hidden_layers = 32


def test_int8_load_kwargs_enable_cpu_offload_and_low_memory_loading() -> None:
    kwargs = build_transformer_load_kwargs("int8", "main", FakeTorch, FakeQuantizationConfig)

    assert kwargs["device_map"] == "auto"
    assert kwargs["dtype"] == "float16"
    assert kwargs["low_cpu_mem_usage"] is True
    assert kwargs["quantization_config"].options == {
        "load_in_8bit": True,
        "llm_int8_enable_fp32_cpu_offload": True,
    }
    assert "torch_dtype" not in kwargs


def test_custom_mistral_map_keeps_overflow_layers_on_cpu_without_disk() -> None:
    device_map = build_mistral_int8_device_map(FakeMistralConfig(), FakeTorch)

    assert device_map["model.layers.0"] == 0
    assert device_map["model.layers.31"] == "cpu"
    assert "disk" not in device_map.values()

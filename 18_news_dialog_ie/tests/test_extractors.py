from extractors import RuleBasedNewsExtractor, extract_json, parse_llm_response
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

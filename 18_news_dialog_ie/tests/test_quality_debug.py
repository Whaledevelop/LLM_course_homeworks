from quality_debug import print_quality_debug
from schemas import ExtractedItem, ExtractionResult


def test_quality_debug_prints_parser_and_normalized_comparison(tmp_path, capsys) -> None:
    gold_path = tmp_path / "gold.csv"
    gold_path.write_text("dialog_id,label,value\none,ORG,OpenAI\none,LOC,New York\n", encoding="utf-8")
    result = ExtractionResult(
        dialog_id="one",
        extractor="qwen",
        entities=[ExtractedItem(label="ORG", value=" openai "), ExtractedItem(label="PERSON", value="Sam Altman")],
        raw_response='{"entities": []}',
    )

    print_quality_debug("qwen-fp16", 1, [result], gold_path, {"one"})

    output = capsys.readouterr().out
    assert "[qwen-fp16] Parse valid: True | error: none" in output
    assert "[qwen-fp16] Raw response:" in output
    assert 'True positives: [{"label": "ORG", "value": "openai"}]' in output
    assert 'False positives: [{"label": "PERSON", "value": "sam altman"}]' in output
    assert 'False negatives: [{"label": "LOC", "value": "new york"}]' in output


def test_quality_debug_reports_subset_without_gold_dialogs(tmp_path, capsys) -> None:
    gold_path = tmp_path / "gold.csv"
    gold_path.write_text("dialog_id,label,value\none,ORG,OpenAI\n", encoding="utf-8")

    print_quality_debug("qwen-int8", 2, [], gold_path, set())

    assert "No gold dialogs are present in this benchmark subset." in capsys.readouterr().out

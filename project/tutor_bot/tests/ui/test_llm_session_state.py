from pathlib import Path
from types import SimpleNamespace

from tutor_bot.generation.llm_response import LlmResponse
from tutor_bot.ui import llm_session_state


def test_persists_default_provider(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        llm_session_state,
        "get_settings",
        lambda: SimpleNamespace(data_dir=tmp_path),
    )

    assert llm_session_state.get_default_provider("ollama") == "ollama"

    llm_session_state.set_default_llm("yandex", "gpt-oss-120b")

    assert llm_session_state.get_default_provider("ollama") == "yandex"
    assert llm_session_state.get_default_model("fallback") == "gpt-oss-120b"
    assert (tmp_path / "llm_preferences.json").read_text(encoding="utf-8") == (
        '{\n  "default_provider": "yandex",\n'
        '  "default_model": "gpt-oss-120b"\n}\n'
    )


def test_records_usage_in_tokens_and_observability(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        llm_session_state,
        "get_settings",
        lambda: SimpleNamespace(
            llm_usage_file=tmp_path / "usage.jsonl",
            history_dir=tmp_path / "history",
        ),
    )
    response = LlmResponse(
        text="answer",
        provider="ollama",
        model="qwen3.5:9b",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )

    llm_session_state.record_usage(response)

    usage = llm_session_state.get_usage()
    event_content = (tmp_path / "history" / "observability_events.jsonl").read_text(
        encoding="utf-8"
    )
    assert usage[0]["total_tokens"] == 15
    assert '"scenario":"llm_generation"' in event_content
    assert '"model":"qwen3.5:9b"' in event_content

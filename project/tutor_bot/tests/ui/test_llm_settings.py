from pathlib import Path
from unittest.mock import Mock

from tutor_bot.config import Settings
from tutor_bot.ui import llm_session_state
from tutor_bot.ui.views import settings_page


def test_vllm_can_be_saved_as_default(monkeypatch, tmp_path: Path) -> None:
    settings = _create_settings(tmp_path, vllm_model="Qwen/Qwen3-8B")
    monkeypatch.setattr(llm_session_state, "get_settings", lambda: settings)

    llm_session_state.set_default_llm("vllm", settings.vllm_model)

    assert llm_session_state.get_default_provider() == "vllm"
    assert llm_session_state.get_default_model("fallback") == "Qwen/Qwen3-8B"


def test_ollama_remains_default_without_preferences(monkeypatch, tmp_path: Path) -> None:
    settings = _create_settings(tmp_path)
    monkeypatch.setattr(llm_session_state, "get_settings", lambda: settings)

    assert llm_session_state.get_default_provider() == "ollama"


def test_vllm_option_uses_configured_model(tmp_path: Path) -> None:
    settings = _create_settings(tmp_path, vllm_model="Qwen/Qwen3-8B")

    options = settings_page._get_llm_options(settings)

    assert options["vllm"] == (
        "vllm",
        "Qwen/Qwen3-8B",
        "vLLM — Qwen/Qwen3-8B",
    )


def test_llms_page_reports_missing_vllm_configuration(monkeypatch, tmp_path: Path) -> None:
    settings = _create_settings(tmp_path)
    error = Mock()
    monkeypatch.setattr(settings_page, "get_settings", lambda: settings)
    monkeypatch.setattr(settings_page, "get_active_provider", lambda _: "vllm")
    monkeypatch.setattr(settings_page, "get_active_model", lambda _: "")
    monkeypatch.setattr(settings_page, "get_default_provider", lambda _: "ollama")
    monkeypatch.setattr(settings_page, "get_default_model", lambda _: settings.ollama_model)
    monkeypatch.setattr(settings_page, "set_active_provider", Mock())
    monkeypatch.setattr(settings_page, "set_active_model", Mock())
    monkeypatch.setattr(settings_page, "render_tokens_statistics_page", Mock())
    monkeypatch.setattr(settings_page.st, "selectbox", lambda *_, **__: "vllm")
    monkeypatch.setattr(settings_page.st, "button", Mock(return_value=False))
    monkeypatch.setattr(settings_page.st, "caption", Mock())
    monkeypatch.setattr(settings_page.st, "success", Mock())
    monkeypatch.setattr(settings_page.st, "error", error)
    monkeypatch.setattr(settings_page.st, "divider", Mock())
    monkeypatch.setattr(settings_page.st, "subheader", Mock())

    settings_page.render_llms_page()

    error.assert_called_once_with("Для vLLM заполните VLLM_API_KEY и VLLM_MODEL в .env.")


def _create_settings(
    tmp_path: Path,
    vllm_model: str = "",
) -> Settings:
    return Settings(
        source_notes_dir=tmp_path,
        project_root=tmp_path,
        VLLM_MODEL=vllm_model,
        _env_file=None,
    )

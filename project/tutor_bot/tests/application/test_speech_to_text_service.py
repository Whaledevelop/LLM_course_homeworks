from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

from tutor_bot.application.speech_to_text_service import (
    SpeechToTextService,
    insert_transcription,
)


class FakeWhisperModel:
    def transcribe(
        self,
        audio: str,
        **kwargs: object,
    ) -> tuple[Iterator[SimpleNamespace], object]:
        assert Path(audio).read_bytes() == b"audio"
        assert kwargs == {
            "language": "ru",
            "vad_filter": True,
            "beam_size": 5,
        }

        return iter([SimpleNamespace(text=" первая "), SimpleNamespace(text="часть")]), object()


def test_transcribes_audio_with_selected_model() -> None:
    selected_models: list[str] = []

    def create_model(model_name: str) -> FakeWhisperModel:
        selected_models.append(model_name)

        return FakeWhisperModel()

    transcription = SpeechToTextService(create_model).transcribe(b"audio", "small")

    assert transcription == "первая часть"
    assert selected_models == ["small"]


def test_inserts_transcription_at_cursor_position() -> None:
    result = insert_transcription(
        "Первая часть текста",
        "голосовая",
        7,
    )

    assert result == "Первая голосовая часть текста"


def test_appends_repeated_transcription_without_merging_words() -> None:
    first_result = insert_transcription("Начало", "середина", 6)
    second_result = insert_transcription(first_result, "конец", len(first_result))

    assert second_result == "Начало середина конец"

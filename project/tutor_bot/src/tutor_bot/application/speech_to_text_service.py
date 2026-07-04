from collections.abc import Callable, Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol


class WhisperSegment(Protocol):
    text: str


class WhisperModel(Protocol):
    def transcribe(
        self,
        audio: str,
        **kwargs: object,
    ) -> tuple[Iterable[WhisperSegment], object]: ...


class SpeechToTextService:
    def __init__(
        self,
        model_factory: Callable[[str], WhisperModel],
    ) -> None:
        self._model_factory = model_factory

    def transcribe(self, audio: bytes, model_name: str) -> str:
        with NamedTemporaryFile(suffix=".webm", delete=False) as audio_file:
            audio_file.write(audio)
            audio_path = Path(audio_file.name)

        try:
            model = self._model_factory(model_name)
            segments, _ = model.transcribe(
                str(audio_path),
                language="ru",
                vad_filter=True,
                beam_size=5,
            )
            transcription = " ".join(
                segment.text.strip()
                for segment in segments
                if segment.text.strip()
            )

            return transcription.strip()
        finally:
            audio_path.unlink(missing_ok=True)


def insert_transcription(text: str, transcription: str, cursor_position: int) -> str:
    insertion = transcription.strip()

    if not insertion:
        return text

    safe_cursor_position = min(max(cursor_position, 0), len(text))
    prefix = text[:safe_cursor_position]
    suffix = text[safe_cursor_position:]
    left_separator = "" if not prefix or prefix[-1].isspace() else " "
    right_separator = "" if not suffix or suffix[0].isspace() else " "

    return f"{prefix}{left_separator}{insertion}{right_separator}{suffix}"

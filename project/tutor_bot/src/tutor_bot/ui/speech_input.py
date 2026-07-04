import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from tutor_bot.application.speech_to_text_service import (
    SpeechToTextService,
    insert_transcription,
)


_SPEECH_MODELS = {
    "small": "Whisper Small — рекомендуется",
    "base": "Whisper Base — быстрее",
    "tiny": "Whisper Tiny — максимально быстро",
}
_speech_recorder = components.declare_component(
    "speech_recorder",
    path=str(Path(__file__).resolve().parent / "components" / "speech_recorder"),
)


@st.cache_resource(show_spinner=False)
def _create_whisper_model(model_name: str):
    from faster_whisper import WhisperModel

    return WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
    )


def render_speech_input(
    text_state_key: str,
    textarea_label: str,
    widget_key: str,
) -> None:
    model_name = st.selectbox(
        "Модель speech-to-text",
        options=list(_SPEECH_MODELS),
        format_func=lambda selected_model: _SPEECH_MODELS[selected_model],
        key=f"{widget_key}-model",
    )
    recording = _speech_recorder(
        textarea_label=textarea_label,
        key=f"{widget_key}-recorder",
        default=None,
    )
    _apply_recording(
        recording,
        model_name,
        text_state_key,
        f"{widget_key}-event",
    )


def _apply_recording(
    recording: dict[str, object] | None,
    model_name: str,
    text_state_key: str,
    event_state_key: str,
) -> None:
    if not recording:
        return

    event_id = str(recording.get("event_id", ""))

    if not event_id or event_id == st.session_state.get(event_state_key):
        return

    st.session_state[event_state_key] = event_id
    error = recording.get("error")

    if error:
        st.error(f"Не удалось записать голос: {error}")

        return

    try:
        audio = base64.b64decode(str(recording["audio"]), validate=True)
        cursor_position = int(recording.get("cursor_position", 0))
        current_text = str(recording.get("question_text", ""))

        with st.spinner("Распознаю речь..."):
            transcription = SpeechToTextService(_create_whisper_model).transcribe(
                audio,
                model_name,
            )

        if not transcription:
            st.error("В записи не удалось распознать речь.")

            return

        st.session_state[text_state_key] = insert_transcription(
            current_text,
            transcription,
            cursor_position,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        st.error(f"Не удалось обработать запись: {error}")

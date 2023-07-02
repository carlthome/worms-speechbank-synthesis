import io

import librosa as lr
from google.cloud import texttospeech


def create_audio(text: str) -> texttospeech.SynthesizeSpeechResponse:
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", name="en-US-Studio-M"
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16, speaking_rate=1.0
    )
    request = {
        "input": input_text,
        "voice": voice,
        "audio_config": audio_config,
    }
    audio_response = client.synthesize_speech(request=request)
    return audio_response


def tweak_audio(audio_response):
    sample_rate = 16000
    with io.BytesIO(audio_response.audio_content) as f:
        waveform, _ = lr.load(f, sr=sample_rate)
    waveform = lr.effects.pitch_shift(waveform, sr=sample_rate, n_steps=12)
    return waveform

from pathlib import Path

import pandas as pd
import vertexai

from .data import get_examples, get_speech_keys
from .llm import create_text
from .tts import create_audio


def main():
    vertexai.init()

    print("Loading data...")
    speech_keys = get_speech_keys()
    examples = get_examples()

    # Load a random personality.
    personalities_path = Path(__file__).resolve().parent / "data" / "personalities.csv"
    personality = pd.read_csv(personalities_path, delimiter=";").sample()

    # Select a random speech key from the included ones.
    speech_key = speech_keys.sample().iloc[0]

    print("Calling LLM API...")
    text_response = create_text(personality, examples, speech_key)

    print("Calling TTS API...")
    audio_response = create_audio(text_response.text)

    # Store audio file in current directory.
    with open("output.mp3", "wb") as f:
        f.write(audio_response.audio_content)


if __name__ == "__main__":
    main()

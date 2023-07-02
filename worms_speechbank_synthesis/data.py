from pathlib import Path

import pandas as pd


def _load_data():
    data_path = Path(__file__).resolve().parent / "data" / "data.txt"
    raw = data_path.read_text()
    rows = pd.Series(raw.splitlines())

    # TODO Make parsing of the wiki text less hackathony. :)
    df = rows[20:-13].str.extract(r"\*(.+)\((.*)\)").dropna()
    df.columns = ["speech_line", "speech_key"]
    return df


def get_speech_keys():
    df = _load_data()

    # Collect most frequent speech keys from the parsed text.
    speech_keys = (
        df.groupby("speech_key")
        .count()
        .nlargest(25, columns="speech_line")
        .reset_index()["speech_key"]
    )
    return speech_keys


def get_examples() -> str:
    df = _load_data()

    # TODO Filter out "speech keys" with only a single occurence (or better yet: manually type out the allowed speech keys in a separate file first).
    rows = df.rename(columns={"speech_key": "input", "speech_line": "output"}).to_dict(
        orient="records"
    )

    # TODO Send semistructured data directly to Vertex AI instead of a string?
    speechbank = ""
    for d in rows:
        speechbank += f"input: {d['input']}\noutput: {d['output']}\n\n"

    return speechbank

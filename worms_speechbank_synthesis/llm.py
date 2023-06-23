from vertexai.language_models import TextGenerationModel


def create_text(personality: dict[str, str], examples: str, speech_key: str) -> str:
    model = TextGenerationModel.from_pretrained("text-bison")

    task_description = """
    You are a game developer for Worms.
    Your are responsible for the speech lines of the worms.
    Given a game event as input, and personality traits and team description as context, you should output a funny speech line with a lot of colorful personality based on the traits.
    Focus a lot on the personality traits and only use the speechbank examples as general inspiration (don't follow them too closely).
    """

    team_name = personality["Team Name"]
    team_description = personality["Team Description"]
    team_traits = personality["Personality Traits"]

    # Construct prompt.
    prompt = f"""
    Task: {task_description}

    Team name: {team_name}

    Team description: {team_description}

    Personality traits: {team_traits}

    Speechbank examples: {examples[:1000]}

    input: {speech_key}\noutput:
    """

    parameters = {
        "temperature": 0.9,
        "max_output_tokens": 64,
        "top_p": 0.8,
        "top_k": 40,
    }
    text_response = model.predict(prompt, **parameters)

    return text_response

from typing import Dict, Any
from json_sanitizer import extract_json_block


def linearize_structured_prompt(structured_prompt: str) -> str:
    """
    Convert structured audio schema into a natural language
    prompt optimized for Stable Audio Open.
    """

    data: Dict[str, Any] = extract_json_block(structured_prompt)
    segments = []

    def add(sentence: str):
        if sentence:
            segments.append(sentence)

    # Core audio description
    if data.get("audio_type"):
        add(f"This is a {data['audio_type']} audio scene")

    if data.get("sound_source"):
        add(f"The sound source is {data['sound_source']}")

    if data.get("sound_event"):
        add(f"The sound event involves {data['sound_event']}")

    if data.get("environment"):
        add(f"The environment is {data['environment']}")

    # Style & temporal attributes
    if data.get("style"):
        add(f"The overall style is {data['style']}")

    if data.get("tempo"):
        add(f"The tempo is {data['tempo']}")

    if data.get("rhythm"):
        add(f"The rhythm is {data['rhythm']}")

    if data.get("mood"):
        add(f"The mood is {data['mood']}")

    # SAO-critical attributes
    if data.get("texture"):
        add(f"The texture is {data['texture']}")

    if data.get("dynamics"):
        add(f"The dynamics are {data['dynamics']}")

    if data.get("spatial"):
        add(f"The spatial characteristics are {data['spatial']}")

    if data.get("structure"):
        add(f"The structure is {data['structure']}")

    # Usage & constraints
    if data.get("production"):
        add(f"The production style is {data['production']}")

    if data.get("use_case"):
        add(f"This audio is intended for {data['use_case']}")

    if data.get("negative"):
        add(f"Avoid the following elements: {data['negative']}")

    return ". ".join(segments) + "."

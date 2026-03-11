from typing import Dict, Any
from json_sanitizer import extract_json_block


def linearize_structured_prompt(structured_prompt: str) -> str:
    """
    Convert structured schema into a single-line Stable Audio Open prompt.
    Prefer a model-provided sao_prompt when available.
    """

    data: Dict[str, Any] = extract_json_block(structured_prompt)

    sao_prompt = str(data.get("sao_prompt", "")).strip()
    if sao_prompt:
        return sao_prompt

    def _get(name: str) -> str:
        value = data.get(name, "")
        return str(value).strip() if value is not None else ""

    merged_instruments = ", ".join(
        part for part in [
            _get("instruments_primary"),
            _get("instruments_supporting"),
            _get("rhythm_components"),
            _get("texture_elements"),
        ]
        if part
    )

    details_merged = ", ".join(
        part for part in [
            _get("use_case"),
            _get("production"),
            _get("looping"),
            _get("duration_hint"),
            _get("details"),
        ]
        if part
    )

    return (
        f"Format: {_get('format')} | "
        f"Genre: {_get('genre')} | "
        f"Sub-genre: {_get('sub_genre')} | "
        f"Instruments: {merged_instruments} | "
        f"Moods: {_get('moods')} | "
        f"Styles: {_get('styles')} | "
        f"Tempo: {_get('tempo')} | "
        f"BPM: {_get('bpm')} | "
        f"Details: {details_merged} | "
        "Quality: high-quality, stereo"
    )

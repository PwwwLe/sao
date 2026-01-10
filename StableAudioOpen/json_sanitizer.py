import json
import re


def extract_json_block(text: str) -> dict:
    """
    Extract the first valid JSON object from LLM output.
    """

    if not text or not text.strip():
        raise ValueError("Empty LLM output")

    text = text.strip()

    # Case 1: wrapped in ```json ``` or ```
    text = re.sub(r"^```(json)?", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()

    # Case 2: extra explanation text, extract {...}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in output")

    json_str = match.group(0)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON after extraction: {e}")

"""Prompt transformation utilities for simulated Qwen2-Audio experiment arms."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTransform:
    """Container describing the intermediate reasoning and final prompt."""

    raw_prompt: str
    reasoning: str | None
    final_prompt: str


STRUCTURED_TEMPLATE = (
    "Audio type: ambience. Scene: {scene}. Sound sources: {sources}. "
    "Acoustic space: immersive stereo environmental field recording. "
    "Temporal behavior: natural evolution over 30 seconds with no abrupt cuts. "
    "Mix notes: balanced foreground and background layers, realistic dynamics, clean ambience."
)

COT_TEMPLATE = (
    "Step 1 - Identify the environment: {scene}. "
    "Step 2 - Extract the most important sound events: {sources}. "
    "Step 3 - Arrange them by depth and continuity so the soundscape feels realistic and loop-friendly. "
    "Step 4 - Emphasize natural ambience, smooth transitions, and believable spatial detail.\n"
    "Final optimized prompt: Create a high-fidelity 30-second ambience recording of {scene} with {sources}, "
    "captured in wide stereo, realistic texture, gentle dynamic variation, and production-ready environmental detail."
)


def _split_prompt(raw_prompt: str) -> tuple[str, str]:
    cleaned = " ".join(raw_prompt.strip().split())
    if not cleaned:
        raise ValueError("raw_prompt must not be empty")

    if " with " in cleaned:
        scene, sources = cleaned.split(" with ", maxsplit=1)
    else:
        scene, sources = cleaned, cleaned

    scene = scene.rstrip(". ")
    sources = sources.rstrip(". ")
    return scene, sources



def generate_structured_prompt(raw_prompt: str) -> str:
    """Expand a raw prompt into a structured acoustic description."""
    scene, sources = _split_prompt(raw_prompt)
    return STRUCTURED_TEMPLATE.format(scene=scene, sources=sources)



def generate_cot_prompt(raw_prompt: str) -> str:
    """Produce a simulated chain-of-thought style reasoning trace plus final prompt."""
    scene, sources = _split_prompt(raw_prompt)
    return COT_TEMPLATE.format(scene=scene, sources=sources)



def build_transform(raw_prompt: str, condition: str) -> PromptTransform:
    """Return the prompt transform for a named experiment condition."""
    if condition == "baseline":
        return PromptTransform(raw_prompt=raw_prompt, reasoning=None, final_prompt=raw_prompt)
    if condition == "structured":
        final_prompt = generate_structured_prompt(raw_prompt)
        return PromptTransform(raw_prompt=raw_prompt, reasoning=None, final_prompt=final_prompt)
    if condition == "cot":
        cot_text = generate_cot_prompt(raw_prompt)
        final_prompt = cot_text.split("Final optimized prompt:", maxsplit=1)[-1].strip()
        return PromptTransform(raw_prompt=raw_prompt, reasoning=cot_text, final_prompt=final_prompt)
    raise ValueError(f"Unsupported condition: {condition}")

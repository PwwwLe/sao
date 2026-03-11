import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import requests

from json_sanitizer import extract_json_block
from prompt_linearizer import linearize_structured_prompt
from sao_utils import SAOGenerator


BASE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = (BASE_DIR / "../experiments").resolve()
LAB_RESULTS_DIR = EXPERIMENT_DIR / "lab_results"
LAB_HISTORY_PATH = EXPERIMENT_DIR / "lab_history.jsonl"
QWEN_SERVICE_URL = os.environ.get(
    "QWEN_SERVICE_URL",
    "http://127.0.0.1:8008/refine_prompt"
)

DEFAULT_PROMPT_SCHEMA = """
{
  "audio_type": "",

    "format": "",
    "genre": "",
    "sub_genre": "",

    "instruments_primary": "",
    "instruments_supporting": "",
    "rhythm_components": "",
    "texture_elements": "",

    "moods": "",
    "styles": "",

    "tempo": "",
    "bpm": "",

    "details": "",
  "production": "",
  "use_case": "",

    "looping": "",
    "duration_hint": "",

    "negative_prompt": "",

    "sao_prompt": ""
}
""".strip()

DEFAULT_SYSTEM_PROMPT = """
You are a professional Stable Audio Open (SAO) prompt engineer and prompt-compiler.

CRITICAL REQUIREMENTS (MUST FOLLOW):
- Output ONLY valid JSON. No explanations, no markdown, no extra keys.
- ALL text values in the output JSON MUST be written in ENGLISH.
- If the user input is not in English, translate it into English internally before filling the JSON.
- Do NOT invent facts. If a field is not inferable from the user input, leave it as an empty string.
- Follow Stable Audio prompt best practices: clear genre/sub-genre, specific moods, explicit instruments, and production characteristics.
- The SAO prompt must be a SINGLE LINE string using the exact field order and separators shown below.

GOAL:
Convert the user's raw request into a structured, controllable SAO input prompt suitable for game audio generation.

AUDIO TYPES:
- "BGM" (music track)
- "AMBIENCE" (environmental soundscape)
- "UI_SFX" (UI/interaction one-shots)
- "FOLEY_SFX" (short sound effects / events)

FORMAT RULES:
- If audio_type is "BGM": format = "Band"
- If audio_type is "AMBIENCE" or "UI_SFX" or "FOLEY_SFX": format = "Solo"

FIELD GUIDELINES (FILL WHEN INFERABLE):
- genre/sub_genre: be specific; sub_genre should refine genre.
- moods: use precise, music-audio-relevant adjectives (e.g., "euphoric", "melancholic", "mystical", "tense", "adventurous").
- styles: include "Video Games" when applicable; add setting tags if explicitly implied (e.g., "Sci-Fi", "Fantasy", "Retro", "High Tech", "Cinematic").
- instruments_*:
    - BGM: separate primary, supporting, rhythm_components, and texture_elements.
    - UI_SFX/FOLEY_SFX: use sound design terms (e.g., "sine ping", "FM blip", "resonant click", "noise burst", "metallic tick").
    - AMBIENCE: describe sound sources (e.g., "field recording rain", "wind", "distant thunder", "room tone") and texture_elements.
- tempo/bpm:
    - If the user provides BPM, copy it.
    - If BPM is not provided, leave bpm empty (do NOT guess).
- looping:
    - For AMBIENCE or loop requests, set looping to "loopable, seamless loop" when explicitly requested.
    - Otherwise leave empty.
- duration_hint:
    - If the user gives a duration, copy it (e.g., "0.5 seconds", "30 seconds loop", "2 minutes").
    - Otherwise leave empty.
- negative_prompt:
    - Only include if the user asks for exclusions or quality constraints; otherwise leave empty.

SAO PROMPT CONSTRUCTION:
Construct "sao_prompt" as a SINGLE LINE using this exact order and separator:
Format: <format> | Genre: <genre> | Sub-genre: <sub_genre> | Instruments: <merged_instruments> | Moods: <moods> | Styles: <styles> | Tempo: <tempo> | BPM: <bpm> | Details: <details_merged> | Quality: high-quality, stereo

Where:
- <merged_instruments> = join non-empty parts in this order:
    instruments_primary; instruments_supporting; rhythm_components; texture_elements
- <details_merged> should compactly include (when non-empty):
    use_case; production; looping; duration_hint; any additional details
- If a field is empty, keep the label but leave the value blank (do NOT remove labels).
    Example: "BPM:  |"

OUTPUT JSON SCHEMA:
{prompt_schema}
""".strip()

SCHEMA_PRESETS = {
    "baseline_v1": DEFAULT_PROMPT_SCHEMA,
    "compact_v2": json.dumps(
        {
            "audio_type": "",
            "format": "",
            "genre": "",
            "sub_genre": "",
            "instruments_primary": "",
            "moods": "",
            "styles": "",
            "tempo": "",
            "bpm": "",
            "details": "",
            "sao_prompt": "",
        },
        ensure_ascii=False,
        indent=2,
    ),
}

MODE_CONFIG = {
    "raw": "Send the raw prompt to Stable Audio Open directly.",
    "structured_nl": "Send the linearized structured prompt to Stable Audio Open.",
    "audio_cot_future": "Reserved mode for a future AudioCoT pipeline. The current demo falls back to the linearized prompt.",
}

_GENERATOR: SAOGenerator | None = None


def ensure_dirs() -> None:
    LAB_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LAB_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_generator() -> SAOGenerator:
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = SAOGenerator()
    return _GENERATOR


def build_system_prompt(system_prompt_template: str, prompt_schema: str) -> str:
    if "{prompt_schema}" in system_prompt_template:
        return system_prompt_template.format(prompt_schema=prompt_schema)
    return system_prompt_template


def call_qwen(
    raw_prompt: str,
    system_prompt_template: str,
    prompt_schema: str,
    max_new_tokens: int,
) -> str:
    payload = {
        "raw_prompt": raw_prompt,
        "max_new_tokens": max_new_tokens,
        "system_prompt": build_system_prompt(system_prompt_template, prompt_schema),
        "prompt_schema": prompt_schema,
    }
    response = requests.post(QWEN_SERVICE_URL, json=payload, timeout=300)
    response.raise_for_status()
    return response.json()["structured_prompt"]


def resolve_final_prompt(mode: str, raw_prompt: str, linearized_prompt: str) -> tuple[str, str]:
    if mode == "raw":
        return raw_prompt, MODE_CONFIG[mode]
    if mode == "structured_nl":
        return linearized_prompt, MODE_CONFIG[mode]
    if mode == "audio_cot_future":
        return linearized_prompt, MODE_CONFIG[mode]
    raise ValueError(f"Unsupported mode: {mode}")


def append_history(entry: dict[str, Any]) -> None:
    ensure_dirs()
    with open(LAB_HISTORY_PATH, "a", encoding="utf-8") as history_file:
        history_file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_history() -> list[dict[str, Any]]:
    if not LAB_HISTORY_PATH.exists():
        return []

    records = []
    with open(LAB_HISTORY_PATH, "r", encoding="utf-8") as history_file:
        for line in history_file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return list(reversed(records))


def history_rows() -> list[list[Any]]:
    rows = []
    for item in load_history():
        rows.append(
            [
                item["run_id"],
                item["created_at"],
                item["mode"],
                item["schema_version"],
                item["seed"],
                item["seconds"],
                item["raw_prompt"],
                item["audio_path"],
            ]
        )
    return rows


def history_choices() -> list[str]:
    return [item["run_id"] for item in load_history()]


def get_history_entry(run_id: str) -> dict[str, Any] | None:
    for item in load_history():
        if item["run_id"] == run_id:
            return item
    return None


def format_steps(steps: list[str]) -> str:
    return "\n".join(f"- {step}" for step in steps)


def run_experiment(
    raw_prompt: str,
    schema_version: str,
    mode: str,
    system_prompt_template: str,
    prompt_schema: str,
    max_new_tokens: int,
    seconds: int,
    seed: int,
    save_history: bool,
    progress=gr.Progress(track_tqdm=False),
):
    steps: list[str] = []

    def snapshot(
        status: str,
        structured_prompt: str = "",
        sanitized_json: str = "",
        linearized_prompt: str = "",
        final_prompt: str = "",
        metadata: dict[str, Any] | None = None,
        audio_path: str | None = None,
        download_path: str | None = None,
    ):
        payload = metadata or {}
        return (
            status,
            format_steps(steps),
            raw_prompt,
            structured_prompt,
            sanitized_json,
            linearized_prompt,
            final_prompt,
            payload,
            audio_path,
            download_path,
        )

    if not raw_prompt or not raw_prompt.strip():
        raise gr.Error("raw_prompt is required")

    ensure_dirs()

    progress(0.02, desc="Preparing experiment")
    steps.append("Received raw prompt and experiment parameters")
    yield snapshot("queued")

    progress(0.12, desc="Calling Qwen service")
    steps.append("Calling Qwen to generate a structured prompt")
    yield snapshot("qwen_processing")
    structured_prompt = call_qwen(
        raw_prompt=raw_prompt,
        system_prompt_template=system_prompt_template,
        prompt_schema=prompt_schema,
        max_new_tokens=max_new_tokens,
    )

    progress(0.28, desc="Sanitizing JSON")
    steps.append("Extracting the JSON block from the Qwen output")
    yield snapshot(
        "parsing",
        structured_prompt=structured_prompt,
    )
    sanitized = extract_json_block(structured_prompt)
    sanitized_json = json.dumps(sanitized, ensure_ascii=False, indent=2)

    progress(0.42, desc="Linearizing prompt")
    steps.append("Linearizing the structured prompt for Stable Audio Open")
    linearized_prompt = linearize_structured_prompt(structured_prompt)
    final_prompt, mode_note = resolve_final_prompt(mode, raw_prompt, linearized_prompt)

    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    audio_path = str(LAB_RESULTS_DIR / f"{run_id}_{mode}.wav")
    metadata = {
        "run_id": run_id,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "mode": mode,
        "mode_note": mode_note,
        "schema_version": schema_version,
        "seed": seed,
        "seconds": seconds,
        "max_new_tokens": max_new_tokens,
        "qwen_service_url": QWEN_SERVICE_URL,
        "raw_prompt": raw_prompt,
        "structured_prompt_raw_text": structured_prompt,
        "structured_prompt": sanitized,
        "linearized_prompt": linearized_prompt,
        "final_prompt": final_prompt,
        "audio_path": audio_path,
    }

    yield snapshot(
        "linearizing",
        structured_prompt=structured_prompt,
        sanitized_json=sanitized_json,
        linearized_prompt=linearized_prompt,
        final_prompt=final_prompt,
        metadata=metadata,
    )

    progress(0.62, desc="Generating audio")
    steps.append("Running Stable Audio Open generation")
    yield snapshot(
        "generating_audio",
        structured_prompt=structured_prompt,
        sanitized_json=sanitized_json,
        linearized_prompt=linearized_prompt,
        final_prompt=final_prompt,
        metadata=metadata,
    )
    generator = get_generator()
    generator.generate(
        prompt=final_prompt,
        seconds=seconds,
        seed=seed,
        out_path=audio_path,
    )

    progress(0.88, desc="Saving outputs")
    steps.append("Saving metadata and output audio")
    if save_history:
        append_history(metadata)

    progress(1.0, desc="Completed")
    steps.append("Experiment completed")
    yield snapshot(
        "completed",
        structured_prompt=structured_prompt,
        sanitized_json=sanitized_json,
        linearized_prompt=linearized_prompt,
        final_prompt=final_prompt,
        metadata=metadata,
        audio_path=audio_path,
        download_path=audio_path,
    )


def apply_schema_preset(schema_version: str) -> str:
    return SCHEMA_PRESETS[schema_version]


def reset_system_prompt() -> str:
    return DEFAULT_SYSTEM_PROMPT


def reset_prompt_schema() -> str:
    return DEFAULT_PROMPT_SCHEMA


def refresh_history_table():
    choices = history_choices()
    return (
        gr.update(value=history_rows()),
        gr.update(choices=choices, value=choices[0] if choices else None),
        gr.update(choices=choices, value=choices[1] if len(choices) > 1 else (choices[0] if choices else None)),
    )


def load_comparison(run_id_a: str, run_id_b: str):
    entry_a = get_history_entry(run_id_a) if run_id_a else None
    entry_b = get_history_entry(run_id_b) if run_id_b else None

    def payload(entry: dict[str, Any] | None) -> dict[str, Any]:
        if entry is None:
            return {}
        return {
            "mode": entry["mode"],
            "schema_version": entry["schema_version"],
            "seed": entry["seed"],
            "seconds": entry["seconds"],
            "raw_prompt": entry["raw_prompt"],
            "linearized_prompt": entry["linearized_prompt"],
            "final_prompt": entry["final_prompt"],
        }

    return (
        payload(entry_a),
        entry_a["audio_path"] if entry_a else None,
        payload(entry_b),
        entry_b["audio_path"] if entry_b else None,
    )


def build_app() -> gr.Blocks:
    ensure_dirs()

    with gr.Blocks(title="SAO Experiment Lab") as demo:
        gr.Markdown(
            """
            # SAO Experiment Lab
            Lightweight Gradio workbench for prompt compilation, intermediate step inspection, audio generation, and A/B comparison.
            """
        )

        with gr.Tab("Run"):
            with gr.Row():
                with gr.Column(scale=1):
                    schema_version = gr.Dropdown(
                        choices=list(SCHEMA_PRESETS.keys()),
                        value="baseline_v1",
                        label="Schema version",
                    )
                    mode = gr.Dropdown(
                        choices=list(MODE_CONFIG.keys()),
                        value="structured_nl",
                        label="Experiment mode",
                    )
                    max_new_tokens = gr.Slider(
                        minimum=64,
                        maximum=1024,
                        step=32,
                        value=512,
                        label="Qwen max_new_tokens",
                    )
                    seconds = gr.Slider(
                        minimum=5,
                        maximum=60,
                        step=1,
                        value=30,
                        label="Audio duration (s)",
                    )
                    seed = gr.Number(value=42, precision=0, label="Seed")
                    save_history = gr.Checkbox(value=True, label="Save experiment history")
                    raw_prompt = gr.Textbox(
                        label="Raw prompt",
                        lines=6,
                        placeholder="Describe the audio you want to generate",
                    )
                    system_prompt_template = gr.Textbox(
                        label="Qwen system prompt template",
                        lines=14,
                        value=DEFAULT_SYSTEM_PROMPT,
                    )
                    prompt_schema = gr.Textbox(
                        label="Prompt schema",
                        lines=18,
                        value=DEFAULT_PROMPT_SCHEMA,
                    )

                    with gr.Row():
                        apply_schema_button = gr.Button("Apply schema preset")
                        reset_system_button = gr.Button("Reset system prompt")
                        reset_schema_button = gr.Button("Reset schema")
                        run_button = gr.Button("Run experiment", variant="primary")

                with gr.Column(scale=1):
                    status = gr.Textbox(label="Current status", interactive=False)
                    step_trace = gr.Markdown(label="Process trace")
                    raw_prompt_echo = gr.Textbox(label="Raw prompt", interactive=False, lines=6)
                    structured_prompt = gr.Textbox(label="Qwen structured prompt", interactive=False, lines=10)
                    sanitized_json = gr.Code(label="Sanitized JSON", language="json", interactive=False)
                    linearized_prompt = gr.Textbox(label="Linearized prompt", interactive=False, lines=8)
                    final_prompt = gr.Textbox(label="Final prompt sent to SAO", interactive=False, lines=8)
                    metadata = gr.JSON(label="Metadata")
                    audio_output = gr.Audio(label="Generated audio", type="filepath")
                    download_file = gr.File(label="Download audio")

            apply_schema_button.click(
                fn=apply_schema_preset,
                inputs=schema_version,
                outputs=prompt_schema,
            )
            reset_system_button.click(fn=reset_system_prompt, outputs=system_prompt_template)
            reset_schema_button.click(fn=reset_prompt_schema, outputs=prompt_schema)
            run_button.click(
                fn=run_experiment,
                inputs=[
                    raw_prompt,
                    schema_version,
                    mode,
                    system_prompt_template,
                    prompt_schema,
                    max_new_tokens,
                    seconds,
                    seed,
                    save_history,
                ],
                outputs=[
                    status,
                    step_trace,
                    raw_prompt_echo,
                    structured_prompt,
                    sanitized_json,
                    linearized_prompt,
                    final_prompt,
                    metadata,
                    audio_output,
                    download_file,
                ],
            )

        with gr.Tab("History and Compare"):
            refresh_button = gr.Button("Refresh history")
            history_table = gr.Dataframe(
                headers=[
                    "run_id",
                    "created_at",
                    "mode",
                    "schema_version",
                    "seed",
                    "seconds",
                    "raw_prompt",
                    "audio_path",
                ],
                value=history_rows(),
                interactive=False,
                wrap=True,
            )

            with gr.Row():
                run_id_a = gr.Dropdown(choices=history_choices(), label="Run A")
                run_id_b = gr.Dropdown(choices=history_choices(), label="Run B")
                compare_button = gr.Button("Load comparison", variant="primary")

            with gr.Row():
                metadata_a = gr.JSON(label="Metadata A")
                metadata_b = gr.JSON(label="Metadata B")

            with gr.Row():
                audio_a = gr.Audio(label="Audio A", type="filepath")
                audio_b = gr.Audio(label="Audio B", type="filepath")

            refresh_button.click(
                fn=refresh_history_table,
                outputs=[history_table, run_id_a, run_id_b],
            )
            compare_button.click(
                fn=load_comparison,
                inputs=[run_id_a, run_id_b],
                outputs=[metadata_a, audio_a, metadata_b, audio_b],
            )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue(default_concurrency_limit=1)
    app.launch(server_name="0.0.0.0", server_port=7860)
"""Unified Gradio frontend containing SAO-only and Qwen-to-SAO experiment pages."""

import json
import os
import random
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
OUTPUT_DIR = EXPERIMENT_DIR / "qwen_sao_outputs"
HISTORY_PATH = EXPERIMENT_DIR / "qwen_sao_history.jsonl"
SAO_OUTPUT_DIR = EXPERIMENT_DIR / "sao_ui_outputs"
SAO_HISTORY_PATH = EXPERIMENT_DIR / "sao_ui_history.jsonl"

QWEN_SERVICE_URL_DEFAULT = os.environ.get(
    "QWEN_SERVICE_URL",
    "http://127.0.0.1:8008/refine_prompt",
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

CRITICAL REQUIREMENTS:
- Output ONLY valid JSON.
- Use ENGLISH for all values.
- Do not add extra keys.
- Leave fields empty if unknown.

OUTPUT JSON SCHEMA:
{prompt_schema}
""".strip()

_DEFAULT_PROMPT = "a clean futuristic UI click with short tail"
_DEFAULT_SAO_PROMPT = (
    "Format: Solo | Genre: Foley | Sub-genre: UI SFX | Instruments: "
    "sine ping, metallic click | Moods: clean, precise | Styles: "
    "Video Games, High Tech | Tempo: Medium | BPM: 120 | Details: "
    "0.5 second one-shot, short decay | Quality: high-quality, stereo"
)
_SAMPLER_OPTIONS = [
    "dpmpp-3m-sde",
    "k-dpm-2",
    "k-dpm-fast",
    "k-heun",
    "k-lms",
]

_GENERATOR: SAOGenerator | None = None


def ensure_dirs() -> None:
    """Create all directories required by both frontend pages."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SAO_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_generator() -> SAOGenerator:
    """Return a cached SAOGenerator shared across requests."""
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = SAOGenerator(device=os.environ.get("SAO_DEVICE", "auto"))
    return _GENERATOR


def build_system_prompt(system_prompt_template: str, prompt_schema: str) -> str:
    """Resolve the Qwen system prompt from template and schema."""
    if "{prompt_schema}" in system_prompt_template:
        return system_prompt_template.format(prompt_schema=prompt_schema)
    return system_prompt_template


def call_qwen(
    qwen_service_url: str,
    raw_prompt: str,
    system_prompt_template: str,
    prompt_schema: str,
    max_new_tokens: int,
) -> str:
    """Call Qwen prompt-compiler service and return structured text output."""
    payload = {
        "raw_prompt": raw_prompt,
        "max_new_tokens": max_new_tokens,
        "system_prompt": build_system_prompt(system_prompt_template, prompt_schema),
        "prompt_schema": prompt_schema,
    }
    resp = requests.post(qwen_service_url, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["structured_prompt"]


def append_history(entry: dict[str, Any]) -> None:
    """Append one Qwen-to-SAO experiment record to JSONL history."""
    ensure_dirs()
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def append_sao_history(entry: dict[str, Any]) -> None:
    """Append one SAO-only run record to JSONL history."""
    ensure_dirs()
    with open(SAO_HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_history() -> list[dict[str, Any]]:
    """Load Qwen experiment history in reverse chronological order."""
    if not HISTORY_PATH.exists():
        return []

    rows: list[dict[str, Any]] = []
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return list(reversed(rows))


def load_sao_history() -> list[dict[str, Any]]:
    """Load SAO-only history in reverse chronological order."""
    if not SAO_HISTORY_PATH.exists():
        return []

    rows: list[dict[str, Any]] = []
    with open(SAO_HISTORY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return list(reversed(rows))


def history_table_rows() -> list[list[Any]]:
    """Build dataframe rows for Qwen experiment history."""
    rows = []
    for item in load_history():
        rows.append(
            [
                item["run_id"],
                item["created_at"],
                item["seed"],
                item["seconds"],
                item["raw_prompt"],
                item["audio_path"],
            ]
        )
    return rows


def history_choices() -> list[str]:
    """Return run identifiers for Qwen experiment history dropdowns."""
    return [item["run_id"] for item in load_history()]


def sao_history_rows() -> list[list[Any]]:
    """Build dataframe rows for SAO-only history."""
    rows = []
    for item in load_sao_history():
        rows.append(
            [
                item["run_id"],
                item["created_at"],
                item["seed"],
                item["seconds"],
                item["steps"],
                item["cfg_scale"],
                item["sampler_type"],
                item["prompt"],
                item["audio_path"],
            ]
        )
    return rows


def sao_history_choices() -> list[str]:
    """Return run identifiers for SAO-only history dropdowns."""
    return [item["run_id"] for item in load_sao_history()]


def get_history_entry(run_id: str) -> dict[str, Any] | None:
    """Find one Qwen experiment record by run id."""
    for item in load_history():
        if item["run_id"] == run_id:
            return item
    return None


def get_sao_history_entry(run_id: str) -> dict[str, Any] | None:
    """Find one SAO-only record by run id."""
    for item in load_sao_history():
        if item["run_id"] == run_id:
            return item
    return None


def refresh_history():
    """Refresh Qwen experiment history table and dropdown."""
    choices = history_choices()
    return (
        gr.update(value=history_table_rows()),
        gr.update(choices=choices, value=choices[0] if choices else None),
    )


def refresh_sao_history():
    """Refresh SAO-only history table and dropdown."""
    choices = sao_history_choices()
    return (
        gr.update(value=sao_history_rows()),
        gr.update(choices=choices, value=choices[0] if choices else None),
    )


def load_history_run(run_id: str):
    """Load selected Qwen experiment record and audio paths."""
    item = get_history_entry(run_id) if run_id else None
    if item is None:
        return {}, None, None
    return item, item["audio_path"], item["audio_path"]


def load_sao_history_run(run_id: str):
    """Load selected SAO-only record and audio paths."""
    item = get_sao_history_entry(run_id) if run_id else None
    if item is None:
        return {}, None, None
    return item, item["audio_path"], item["audio_path"]


def random_seed_value() -> int:
    """Generate a random positive 32-bit seed value."""
    return random.randint(1, 2**31 - 1)


def run_sao_generation(
    prompt: str,
    seconds: int,
    seed: int,
    steps: int,
    cfg_scale: float,
    sigma_min: float,
    sigma_max: float,
    sampler_type: str,
    save_history: bool,
    progress=gr.Progress(track_tqdm=False),
):
    """Run standalone SAO generation workflow and emit UI outputs."""
    if not prompt or not prompt.strip():
        raise gr.Error("Prompt is required")

    ensure_dirs()

    progress(0.05, desc="Preparing")
    status = "preparing"

    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    audio_path = str(SAO_OUTPUT_DIR / f"{run_id}.wav")

    progress(0.2, desc="Loading generator")
    status = "loading_model"
    generator = get_generator()

    progress(0.35, desc="Generating audio")
    status = "generating_audio"
    generator.generate(
        prompt=prompt,
        seconds=int(seconds),
        seed=int(seed),
        out_path=audio_path,
        steps=int(steps),
        cfg_scale=float(cfg_scale),
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
        sampler_type=sampler_type,
    )

    progress(0.9, desc="Saving metadata")
    status = "saving"

    metadata = {
        "run_id": run_id,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "prompt": prompt,
        "seconds": int(seconds),
        "seed": int(seed),
        "steps": int(steps),
        "cfg_scale": float(cfg_scale),
        "sigma_min": float(sigma_min),
        "sigma_max": float(sigma_max),
        "sampler_type": sampler_type,
        "audio_path": audio_path,
        "storage_dir": str(SAO_OUTPUT_DIR),
    }

    if save_history:
        append_sao_history(metadata)

    progress(1.0, desc="Completed")
    status = "completed"

    return (
        status,
        metadata,
        audio_path,
        audio_path,
        audio_path,
        str(SAO_OUTPUT_DIR),
        gr.update(value=sao_history_rows()),
        gr.update(choices=sao_history_choices(), value=run_id),
    )


def run_experiment(
    qwen_service_url: str,
    raw_prompt: str,
    system_prompt_template: str,
    prompt_schema: str,
    max_new_tokens: int,
    seconds: int,
    seed: int,
    save_history: bool,
    progress=gr.Progress(track_tqdm=False),
):
    """Run end-to-end Qwen-to-SAO experiment with progress snapshots."""
    steps: list[str] = []

    def snapshot(
        status: str,
        structured_prompt: str = "",
        sanitized_json: str = "",
        linearized_prompt: str = "",
        metadata: dict[str, Any] | None = None,
        audio_path: str | None = None,
        download_path: str | None = None,
    ):
        return (
            status,
            "\n".join(f"- {s}" for s in steps),
            raw_prompt,
            structured_prompt,
            sanitized_json,
            linearized_prompt,
            metadata or {},
            audio_path,
            download_path,
            str(OUTPUT_DIR),
        )

    if not raw_prompt or not raw_prompt.strip():
        raise gr.Error("raw_prompt is required")

    ensure_dirs()

    progress(0.05, desc="Qwen processing")
    steps.append("Calling Qwen2Audio server")
    yield snapshot("qwen_processing")

    try:
        structured_prompt = call_qwen(
            qwen_service_url=qwen_service_url,
            raw_prompt=raw_prompt,
            system_prompt_template=system_prompt_template,
            prompt_schema=prompt_schema,
            max_new_tokens=max_new_tokens,
        )
    except requests.RequestException as exc:
        raise gr.Error(f"Qwen request failed: {exc}")

    progress(0.25, desc="Parsing JSON")
    steps.append("Sanitizing JSON output")
    yield snapshot("parsing", structured_prompt=structured_prompt)

    try:
        sanitized = extract_json_block(structured_prompt)
    except Exception as exc:
        raise gr.Error(f"JSON sanitize failed: {exc}")

    sanitized_json = json.dumps(sanitized, ensure_ascii=False, indent=2)

    progress(0.4, desc="Linearizing prompt")
    steps.append("Linearizing structured prompt for SAO")
    linearized_prompt = linearize_structured_prompt(structured_prompt)
    yield snapshot(
        "linearizing",
        structured_prompt=structured_prompt,
        sanitized_json=sanitized_json,
        linearized_prompt=linearized_prompt,
    )

    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    audio_path = str(OUTPUT_DIR / f"{run_id}.wav")

    progress(0.6, desc="Generating audio")
    steps.append("Generating audio with Stable Audio Open")
    yield snapshot(
        "generating_audio",
        structured_prompt=structured_prompt,
        sanitized_json=sanitized_json,
        linearized_prompt=linearized_prompt,
    )

    generator = get_generator()
    generator.generate(
        prompt=linearized_prompt,
        seconds=int(seconds),
        seed=int(seed),
        out_path=audio_path,
    )

    progress(0.9, desc="Saving result")
    steps.append("Saving audio and experiment record")

    metadata = {
        "run_id": run_id,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "qwen_service_url": qwen_service_url,
        "raw_prompt": raw_prompt,
        "structured_prompt_raw_text": structured_prompt,
        "structured_prompt": sanitized,
        "linearized_prompt": linearized_prompt,
        "seconds": int(seconds),
        "seed": int(seed),
        "max_new_tokens": int(max_new_tokens),
        "audio_path": audio_path,
        "storage_dir": str(OUTPUT_DIR),
    }

    if save_history:
        append_history(metadata)

    progress(1.0, desc="Completed")
    steps.append("Completed")
    yield snapshot(
        "completed",
        structured_prompt=structured_prompt,
        sanitized_json=sanitized_json,
        linearized_prompt=linearized_prompt,
        metadata=metadata,
        audio_path=audio_path,
        download_path=audio_path,
    )


def build_app() -> gr.Blocks:
    """Build and return the unified multi-page Gradio app."""
    ensure_dirs()

    with gr.Blocks(title="Qwen + SAO Experiment Lab") as demo:
        gr.Markdown(
            """
            # Qwen + SAO Experiment Lab
            One service with two pages:
            1) SAO-only quick generation
            2) Qwen-to-SAO experiment workflow
            """
        )

        with gr.Tab("SAO Simple"):
            with gr.Tab("Run"):
                with gr.Row():
                    with gr.Column(scale=1):
                        sao_prompt = gr.Textbox(
                            label="Prompt",
                            lines=8,
                            value=_DEFAULT_SAO_PROMPT,
                            placeholder="Describe your target audio prompt",
                        )
                        sao_seconds = gr.Slider(minimum=1, maximum=60, step=1, value=30, label="Duration (seconds)")
                        sao_seed = gr.Number(value=42, precision=0, label="Seed")
                        sao_random_seed_button = gr.Button("Random Seed")
                        sao_steps = gr.Slider(minimum=10, maximum=200, step=5, value=100, label="Steps")
                        sao_cfg_scale = gr.Slider(minimum=1.0, maximum=12.0, step=0.5, value=7.0, label="CFG Scale")
                        sao_sigma_min = gr.Slider(minimum=0.05, maximum=1.0, step=0.05, value=0.3, label="Sigma Min")
                        sao_sigma_max = gr.Slider(minimum=100.0, maximum=1000.0, step=25.0, value=500.0, label="Sigma Max")
                        sao_sampler_type = gr.Dropdown(
                            choices=_SAMPLER_OPTIONS,
                            value="dpmpp-3m-sde",
                            label="Sampler",
                        )
                        sao_save_history = gr.Checkbox(value=True, label="Save history")
                        sao_generate_button = gr.Button("Generate Audio", variant="primary")

                    with gr.Column(scale=1):
                        sao_status = gr.Textbox(label="Status", interactive=False)
                        sao_metadata = gr.JSON(label="Run Metadata")
                        sao_audio_output = gr.Audio(label="Generated Audio", type="filepath")
                        sao_download_file = gr.File(label="Download Audio")
                        sao_output_path = gr.Textbox(label="Output File Path", interactive=False)
                        sao_storage_dir = gr.Textbox(label="Storage Directory", value=str(SAO_OUTPUT_DIR), interactive=False)

                sao_random_seed_button.click(fn=random_seed_value, outputs=sao_seed)

            with gr.Tab("History"):
                sao_refresh_button = gr.Button("Refresh History")
                sao_history_table = gr.Dataframe(
                    headers=[
                        "run_id",
                        "created_at",
                        "seed",
                        "seconds",
                        "steps",
                        "cfg_scale",
                        "sampler",
                        "prompt",
                        "audio_path",
                    ],
                    value=sao_history_rows(),
                    interactive=False,
                    wrap=True,
                )
                sao_history_run_id = gr.Dropdown(choices=sao_history_choices(), label="Select Run")
                sao_load_history_button = gr.Button("Load Selected Run")
                sao_history_metadata = gr.JSON(label="History Metadata")
                sao_history_audio = gr.Audio(label="History Audio", type="filepath")
                sao_history_file = gr.File(label="History Audio File")

                sao_refresh_button.click(
                    fn=refresh_sao_history,
                    outputs=[sao_history_table, sao_history_run_id],
                )

                sao_load_history_button.click(
                    fn=load_sao_history_run,
                    inputs=[sao_history_run_id],
                    outputs=[sao_history_metadata, sao_history_audio, sao_history_file],
                )

                # connect generation output to history widgets after they are defined
                sao_generate_button.click(
                    fn=run_sao_generation,
                    inputs=[
                        sao_prompt,
                        sao_seconds,
                        sao_seed,
                        sao_steps,
                        sao_cfg_scale,
                        sao_sigma_min,
                        sao_sigma_max,
                        sao_sampler_type,
                        sao_save_history,
                    ],
                    outputs=[
                        sao_status,
                        sao_metadata,
                        sao_audio_output,
                        sao_download_file,
                        sao_output_path,
                        sao_storage_dir,
                        sao_history_table,
                        sao_history_run_id,
                    ],
                )

        with gr.Tab("Qwen + SAO Experiment"):
            with gr.Tab("Run"):
                with gr.Row():
                    with gr.Column(scale=1):
                        qwen_service_url = gr.Textbox(
                            label="Qwen Service URL",
                            value=QWEN_SERVICE_URL_DEFAULT,
                        )
                        raw_prompt = gr.Textbox(
                            label="Raw Prompt",
                            value=_DEFAULT_PROMPT,
                            lines=5,
                        )
                        max_new_tokens = gr.Slider(
                            minimum=64,
                            maximum=1024,
                            step=32,
                            value=512,
                            label="Qwen max_new_tokens",
                        )
                        seconds = gr.Slider(
                            minimum=1,
                            maximum=60,
                            step=1,
                            value=30,
                            label="SAO Duration (seconds)",
                        )
                        seed = gr.Number(value=42, precision=0, label="Seed")
                        save_history = gr.Checkbox(value=True, label="Save History")
                        system_prompt_template = gr.Textbox(
                            label="Qwen System Prompt",
                            lines=16,
                            value=DEFAULT_SYSTEM_PROMPT,
                        )
                        prompt_schema = gr.Textbox(
                            label="Prompt Schema",
                            lines=14,
                            value=DEFAULT_PROMPT_SCHEMA,
                        )
                        run_button = gr.Button("Run Experiment", variant="primary")

                    with gr.Column(scale=1):
                        status = gr.Textbox(label="Current Status", interactive=False)
                        process_steps = gr.Markdown(label="Process Steps")
                        raw_prompt_echo = gr.Textbox(label="Raw Prompt", interactive=False, lines=5)
                        structured_prompt = gr.Textbox(label="Qwen Structured Prompt", interactive=False, lines=8)
                        sanitized_json = gr.Code(label="Sanitized JSON", language="json", interactive=False)
                        linearized_prompt = gr.Textbox(label="Linearized Prompt (to SAO)", interactive=False, lines=8)
                        metadata = gr.JSON(label="Metadata")
                        generated_audio = gr.Audio(label="Generated Audio", type="filepath")
                        download_file = gr.File(label="Download Audio")
                        output_dir = gr.Textbox(label="Storage Directory", interactive=False, value=str(OUTPUT_DIR))

                run_button.click(
                    fn=run_experiment,
                    inputs=[
                        qwen_service_url,
                        raw_prompt,
                        system_prompt_template,
                        prompt_schema,
                        max_new_tokens,
                        seconds,
                        seed,
                        save_history,
                    ],
                    outputs=[
                        status,
                        process_steps,
                        raw_prompt_echo,
                        structured_prompt,
                        sanitized_json,
                        linearized_prompt,
                        metadata,
                        generated_audio,
                        download_file,
                        output_dir,
                    ],
                )

            with gr.Tab("History"):
                refresh_button = gr.Button("Refresh")
                history_table = gr.Dataframe(
                    headers=[
                        "run_id",
                        "created_at",
                        "seed",
                        "seconds",
                        "raw_prompt",
                        "audio_path",
                    ],
                    value=history_table_rows(),
                    interactive=False,
                    wrap=True,
                )
                history_run_id = gr.Dropdown(label="Select Run", choices=history_choices())
                load_button = gr.Button("Load Selected Run")
                history_metadata = gr.JSON(label="History Metadata")
                history_audio = gr.Audio(label="History Audio", type="filepath")
                history_file = gr.File(label="History File")

                refresh_button.click(
                    fn=refresh_history,
                    outputs=[history_table, history_run_id],
                )
                load_button.click(
                    fn=load_history_run,
                    inputs=[history_run_id],
                    outputs=[history_metadata, history_audio, history_file],
                )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue(default_concurrency_limit=1)
    app.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", os.environ.get("GRADIO_EXPERIMENT_PORT", "7860"))),
    )

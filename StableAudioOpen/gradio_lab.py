"""Gradio frontend for standalone SAO generation and history browsing."""

import json
import os
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr

from sao_utils import SAOGenerator


BASE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = (BASE_DIR / "../experiments").resolve()
OUTPUT_DIR = EXPERIMENT_DIR / "sao_ui_outputs"
HISTORY_PATH = EXPERIMENT_DIR / "sao_ui_history.jsonl"

DEFAULT_PROMPT = (
    "Format: Solo | Genre: Foley | Sub-genre: UI SFX | Instruments: "
    "sine ping, metallic click | Moods: clean, precise | Styles: "
    "Video Games, High Tech | Tempo: Medium | BPM: 120 | Details: "
    "0.5 second one-shot, short decay | Quality: high-quality, stereo"
)

SAMPLER_OPTIONS = [
    "dpmpp-3m-sde",
    "k-dpm-2",
    "k-dpm-fast",
    "k-heun",
    "k-lms",
]

_GENERATOR: SAOGenerator | None = None


def ensure_dirs() -> None:
    """Create output and history directories if they do not exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_generator() -> SAOGenerator:
    """Return a cached SAOGenerator instance."""
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = SAOGenerator(device=os.environ.get("SAO_DEVICE", "auto"))
    return _GENERATOR


def append_history(entry: dict[str, Any]) -> None:
    """Append one generation record to the JSONL history file."""
    ensure_dirs()
    with open(HISTORY_PATH, "a", encoding="utf-8") as history_file:
        history_file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_history() -> list[dict[str, Any]]:
    """Load history entries in reverse chronological order."""
    if not HISTORY_PATH.exists():
        return []

    records: list[dict[str, Any]] = []
    with open(HISTORY_PATH, "r", encoding="utf-8") as history_file:
        for line in history_file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return list(reversed(records))


def history_rows() -> list[list[Any]]:
    """Convert history records to rows for the Gradio dataframe."""
    rows = []
    for item in load_history():
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


def history_choices() -> list[str]:
    """Return run_id options for history dropdown widgets."""
    return [item["run_id"] for item in load_history()]


def get_history_entry(run_id: str) -> dict[str, Any] | None:
    """Find one history record by run identifier."""
    for item in load_history():
        if item["run_id"] == run_id:
            return item
    return None


def random_seed_value() -> int:
    """Generate a random positive 32-bit seed value."""
    return random.randint(1, 2**31 - 1)


def refresh_history_view():
    """Refresh history table and dropdown selection."""
    choices = history_choices()
    first = choices[0] if choices else None
    return (
        gr.update(value=history_rows()),
        gr.update(choices=choices, value=first),
    )


def load_history_item(run_id: str):
    """Load selected history metadata and associated audio paths."""
    item = get_history_entry(run_id) if run_id else None
    if item is None:
        return {}, None, None
    return item, item["audio_path"], item["audio_path"]


def run_generation(
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
    """Generate one audio sample and optionally persist run history."""
    if not prompt or not prompt.strip():
        raise gr.Error("Prompt is required")

    ensure_dirs()

    progress(0.05, desc="Preparing")
    status = "preparing"

    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    audio_path = str(OUTPUT_DIR / f"{run_id}.wav")

    progress(0.2, desc="Loading generator")
    status = "loading_model"
    generator = get_generator()

    progress(0.35, desc="Generating audio")
    status = "generating_audio"
    generator.generate(
        prompt=prompt,
        seconds=seconds,
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
        "storage_dir": str(OUTPUT_DIR),
    }

    if save_history:
        append_history(metadata)

    progress(1.0, desc="Completed")
    status = "completed"

    return (
        status,
        metadata,
        audio_path,
        audio_path,
        audio_path,
        str(OUTPUT_DIR),
        gr.update(value=history_rows()),
        gr.update(choices=history_choices(), value=run_id),
    )


def build_app() -> gr.Blocks:
    """Build and return the standalone SAO Gradio application."""
    ensure_dirs()

    with gr.Blocks(title="SAO Simple Generator") as demo:
        gr.Markdown(
            """
            # SAO Simple Generator
            Minimal front-end for Stable Audio Open generation.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    lines=8,
                    value=DEFAULT_PROMPT,
                    placeholder="Describe your target audio prompt",
                )

                seconds = gr.Slider(minimum=1, maximum=60, step=1, value=30, label="Duration (seconds)")
                seed = gr.Number(value=42, precision=0, label="Seed")
                random_seed_button = gr.Button("Random Seed")

                steps = gr.Slider(minimum=10, maximum=200, step=5, value=100, label="Steps")
                cfg_scale = gr.Slider(minimum=1.0, maximum=12.0, step=0.5, value=7.0, label="CFG Scale")
                sigma_min = gr.Slider(minimum=0.05, maximum=1.0, step=0.05, value=0.3, label="Sigma Min")
                sigma_max = gr.Slider(minimum=100.0, maximum=1000.0, step=25.0, value=500.0, label="Sigma Max")
                sampler_type = gr.Dropdown(
                    choices=SAMPLER_OPTIONS,
                    value="dpmpp-3m-sde",
                    label="Sampler",
                )

                save_history = gr.Checkbox(value=True, label="Save history")
                generate_button = gr.Button("Generate Audio", variant="primary")

            with gr.Column(scale=1):
                status = gr.Textbox(label="Status", interactive=False)
                metadata = gr.JSON(label="Run Metadata")
                audio_output = gr.Audio(label="Generated Audio", type="filepath")
                download_file = gr.File(label="Download Audio")
                output_path = gr.Textbox(label="Output File Path", interactive=False)
                storage_dir = gr.Textbox(label="Storage Directory", value=str(OUTPUT_DIR), interactive=False)

        with gr.Tab("History"):
            refresh_button = gr.Button("Refresh History")
            history_table = gr.Dataframe(
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
                value=history_rows(),
                interactive=False,
                wrap=True,
            )
            history_run_id = gr.Dropdown(choices=history_choices(), label="Select Run")
            load_history_button = gr.Button("Load Selected Run")
            history_metadata = gr.JSON(label="History Metadata")
            history_audio = gr.Audio(label="History Audio", type="filepath")
            history_file = gr.File(label="History Audio File")

        random_seed_button.click(fn=random_seed_value, outputs=seed)

        generate_button.click(
            fn=run_generation,
            inputs=[
                prompt,
                seconds,
                seed,
                steps,
                cfg_scale,
                sigma_min,
                sigma_max,
                sampler_type,
                save_history,
            ],
            outputs=[
                status,
                metadata,
                audio_output,
                download_file,
                output_path,
                storage_dir,
                history_table,
                history_run_id,
            ],
        )

        refresh_button.click(
            fn=refresh_history_view,
            outputs=[history_table, history_run_id],
        )

        load_history_button.click(
            fn=load_history_item,
            inputs=[history_run_id],
            outputs=[history_metadata, history_audio, history_file],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue(default_concurrency_limit=1)
    app.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )

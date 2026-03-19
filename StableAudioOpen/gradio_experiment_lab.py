"""Unified Gradio frontend containing SAO-only and Qwen-to-SAO experiment pages."""

import json
import os
import random
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
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
SAO_BATCH_EXPORT_DIR = EXPERIMENT_DIR / "sao_batch_exports"

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
_TASK_FILTER_OPTIONS = ["all", "queued", "running", "completed", "failed", "cancelled"]

_GENERATOR: SAOGenerator | None = None
_GENERATOR_LOCAL = threading.local()
_BATCH_MANAGER: "BatchTaskManager | None" = None


def utc_now() -> datetime:
    """Return timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def isoformat_utc(dt: datetime | None) -> str | None:
    """Convert UTC datetime to ISO8601 text."""
    if dt is None:
        return None
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def format_elapsed_seconds(started_at: datetime | None, ended_at: datetime | None) -> str:
    """Return elapsed seconds as text when start/end timestamps exist."""
    if started_at is None:
        return ""
    finished_at = ended_at or utc_now()
    return f"{(finished_at - started_at).total_seconds():.2f}s"


def sanitize_text(value: Any) -> str:
    """Escape arbitrary values for simple HTML output."""
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def audio_url(path: str | None) -> str | None:
    """Return Gradio-served file URL for a saved audio path."""
    if not path:
        return None
    return f"/gradio_api/file={path}"


def ensure_dirs() -> None:
    """Create all directories required by both frontend pages."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SAO_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAO_BATCH_EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def get_generator() -> SAOGenerator:
    """Return a cached SAOGenerator shared across single-run requests."""
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = SAOGenerator(device=os.environ.get("SAO_DEVICE", "auto"))
    return _GENERATOR


def get_worker_generator() -> SAOGenerator:
    """Return a thread-local SAOGenerator for batch worker threads."""
    generator = getattr(_GENERATOR_LOCAL, "generator", None)
    if generator is None:
        generator = SAOGenerator(device=os.environ.get("SAO_DEVICE", "auto"))
        _GENERATOR_LOCAL.generator = generator
    return generator


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


class BatchTaskManager:
    """In-memory batch scheduler for multi-prompt SAO experiments."""

    def __init__(self, max_workers: int):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sao-batch")
        self.lock = threading.RLock()
        self.tasks: dict[str, dict[str, Any]] = {}
        self.batches: dict[str, list[str]] = {}

    def submit_batch(
        self,
        prompts: list[str],
        seconds: int,
        seed: int,
        steps: int,
        cfg_scale: float,
        sigma_min: float,
        sigma_max: float,
        sampler_type: str,
        experiment_group: str,
        note: str,
        save_history: bool,
    ) -> str:
        batch_id = utc_now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        submitted_at = utc_now()
        task_ids: list[str] = []

        for index, prompt in enumerate(prompts, start=1):
            task_id = f"{batch_id}-t{index:03d}"
            audio_path = str(SAO_OUTPUT_DIR / batch_id / f"{task_id}.wav")
            task = {
                "task_id": task_id,
                "batch_id": batch_id,
                "prompt": prompt,
                "experiment_group": experiment_group,
                "note": note,
                "status": "queued",
                "submitted_at": submitted_at,
                "started_at": None,
                "ended_at": None,
                "duration_seconds": seconds,
                "seed": int(seed) + (index - 1),
                "steps": int(steps),
                "cfg_scale": float(cfg_scale),
                "sigma_min": float(sigma_min),
                "sigma_max": float(sigma_max),
                "sampler_type": sampler_type,
                "save_history": bool(save_history),
                "audio_path": audio_path,
                "error": "",
                "result_ready": False,
            }
            with self.lock:
                self.tasks[task_id] = task
                self.batches.setdefault(batch_id, []).append(task_id)
            task_ids.append(task_id)
            self.executor.submit(self._run_task, task_id)

        return batch_id

    def _run_task(self, task_id: str) -> None:
        with self.lock:
            task = self.tasks.get(task_id)
            if task is None or task["status"] != "queued":
                return
            task["status"] = "running"
            task["started_at"] = utc_now()
            task["error"] = ""

        try:
            output_path = Path(task["audio_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            generator = get_worker_generator()
            generator.generate(
                prompt=task["prompt"],
                seconds=task["duration_seconds"],
                seed=task["seed"],
                out_path=str(output_path),
                steps=task["steps"],
                cfg_scale=task["cfg_scale"],
                sigma_min=task["sigma_min"],
                sigma_max=task["sigma_max"],
                sampler_type=task["sampler_type"],
            )
        except Exception as exc:
            with self.lock:
                latest = self.tasks.get(task_id)
                if latest is None:
                    return
                latest["status"] = "failed"
                latest["ended_at"] = utc_now()
                latest["error"] = str(exc)
            return

        with self.lock:
            latest = self.tasks.get(task_id)
            if latest is None:
                return
            latest["status"] = "completed"
            latest["ended_at"] = utc_now()
            latest["result_ready"] = True
            history_entry = self._task_metadata(latest)

        if history_entry["save_history"]:
            append_sao_history(history_entry)

    def _task_metadata(self, task: dict[str, Any]) -> dict[str, Any]:
        return {
            "run_id": task["task_id"],
            "task_id": task["task_id"],
            "batch_id": task["batch_id"],
            "created_at": isoformat_utc(task["submitted_at"]),
            "submitted_at": isoformat_utc(task["submitted_at"]),
            "started_at": isoformat_utc(task["started_at"]),
            "ended_at": isoformat_utc(task["ended_at"]),
            "elapsed": format_elapsed_seconds(task["started_at"], task["ended_at"]),
            "prompt": task["prompt"],
            "experiment_group": task["experiment_group"],
            "note": task["note"],
            "status": task["status"],
            "seconds": task["duration_seconds"],
            "seed": task["seed"],
            "steps": task["steps"],
            "cfg_scale": task["cfg_scale"],
            "sigma_min": task["sigma_min"],
            "sigma_max": task["sigma_max"],
            "sampler_type": task["sampler_type"],
            "audio_path": task["audio_path"],
            "storage_dir": str(Path(task["audio_path"]).parent),
            "error": task["error"],
            "save_history": task["save_history"],
        }

    def batch_choices(self) -> list[str]:
        with self.lock:
            return sorted(self.batches.keys(), reverse=True)

    def batch_summary(self, batch_id: str | None) -> dict[str, Any]:
        tasks = self.tasks_for_batch(batch_id)
        counts: dict[str, int] = {status: 0 for status in _TASK_FILTER_OPTIONS if status != "all"}
        for task in tasks:
            counts[task["status"]] = counts.get(task["status"], 0) + 1
        return {
            "batch_id": batch_id or "",
            "total": len(tasks),
            "queued": counts.get("queued", 0),
            "running": counts.get("running", 0),
            "completed": counts.get("completed", 0),
            "failed": counts.get("failed", 0),
            "cancelled": counts.get("cancelled", 0),
            "max_workers": self.max_workers,
        }

    def tasks_for_batch(self, batch_id: str | None, status_filter: str = "all") -> list[dict[str, Any]]:
        with self.lock:
            task_ids = list(self.batches.get(batch_id, [])) if batch_id else []
            tasks = [dict(self.tasks[task_id]) for task_id in task_ids if task_id in self.tasks]
        tasks.sort(key=lambda item: (item["submitted_at"], item["task_id"]))
        if status_filter != "all":
            tasks = [item for item in tasks if item["status"] == status_filter]
        return tasks

    def cancel_waiting(self, batch_id: str | None) -> int:
        changed = 0
        with self.lock:
            for task_id in self.batches.get(batch_id, []):
                task = self.tasks.get(task_id)
                if task and task["status"] == "queued":
                    task["status"] = "cancelled"
                    task["ended_at"] = utc_now()
                    task["error"] = "Cancelled before execution"
                    changed += 1
        return changed

    def clear_completed(self, batch_id: str | None) -> int:
        removed = 0
        with self.lock:
            task_ids = self.batches.get(batch_id, [])
            kept: list[str] = []
            for task_id in task_ids:
                task = self.tasks.get(task_id)
                if task and task["status"] == "completed":
                    self.tasks.pop(task_id, None)
                    removed += 1
                else:
                    kept.append(task_id)
            if batch_id in self.batches:
                self.batches[batch_id] = kept
        return removed

    def retry_failed(self, batch_id: str | None) -> int:
        failed_tasks = self.tasks_for_batch(batch_id, status_filter="failed")
        if not failed_tasks:
            return 0
        for task in failed_tasks:
            retry_task_id = f"{task['task_id']}-retry-{uuid.uuid4().hex[:4]}"
            retry_audio_path = str(Path(task["audio_path"]).with_name(f"{retry_task_id}.wav"))
            new_task = {
                **task,
                "task_id": retry_task_id,
                "status": "queued",
                "submitted_at": utc_now(),
                "started_at": None,
                "ended_at": None,
                "audio_path": retry_audio_path,
                "error": "",
                "result_ready": False,
            }
            with self.lock:
                self.tasks[retry_task_id] = new_task
                self.batches.setdefault(task["batch_id"], []).append(retry_task_id)
            self.executor.submit(self._run_task, retry_task_id)
        return len(failed_tasks)

    def export_batch(self, batch_id: str | None) -> str | None:
        if not batch_id:
            return None
        tasks = [self._task_metadata(task) for task in self.tasks_for_batch(batch_id)]
        if not tasks:
            return None
        export_path = SAO_BATCH_EXPORT_DIR / f"{batch_id}.json"
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with open(export_path, "w", encoding="utf-8") as export_file:
            json.dump({"batch_id": batch_id, "tasks": tasks}, export_file, ensure_ascii=False, indent=2)
        return str(export_path)


def get_batch_manager() -> BatchTaskManager:
    """Return the singleton multi-task batch manager."""
    global _BATCH_MANAGER
    if _BATCH_MANAGER is None:
        max_workers = max(1, int(os.environ.get("SAO_BATCH_MAX_WORKERS", "2")))
        _BATCH_MANAGER = BatchTaskManager(max_workers=max_workers)
    return _BATCH_MANAGER


def parse_prompt_lines(prompt_text: str) -> list[str]:
    """Parse multi-line prompt input, keeping one non-empty prompt per line."""
    prompts = [line.strip() for line in prompt_text.splitlines() if line.strip()]
    if not prompts:
        raise gr.Error("Please provide at least one prompt (one prompt per line).")
    return prompts


def experiment_table_value(raw_text: str, group: str, note: str) -> list[list[str]]:
    """Render prompt input into a lightweight preview table."""
    prompts = [line.strip() for line in raw_text.splitlines() if line.strip()]
    rows = []
    for index, prompt in enumerate(prompts, start=1):
        rows.append([str(index), prompt, group, note])
    return rows


def build_task_rows(tasks: list[dict[str, Any]]) -> list[list[Any]]:
    """Convert task metadata to dataframe rows."""
    rows = []
    for task in tasks:
        rows.append(
            [
                task["task_id"],
                task["prompt"],
                task["status"],
                isoformat_utc(task["submitted_at"]),
                isoformat_utc(task["started_at"]),
                isoformat_utc(task["ended_at"]),
                format_elapsed_seconds(task["started_at"], task["ended_at"]),
                task["audio_path"] if task["status"] == "completed" else "",
                task["error"],
                task["experiment_group"],
                task["note"],
            ]
        )
    return rows


def build_results_html(tasks: list[dict[str, Any]]) -> str:
    """Render completed task results as comparable audio cards."""
    completed = [task for task in tasks if task["status"] == "completed" and task["audio_path"]]
    if not completed:
        return (
            "<div class='result-empty'>No completed results yet. "
            "Submitted tasks will appear here as soon as each audio file is ready.</div>"
        )

    cards = []
    for task in completed:
        src = audio_url(task["audio_path"])
        cards.append(
            "<div class='result-card'>"
            f"<div class='result-card-header'><strong>{sanitize_text(task['task_id'])}</strong>"
            f" <span class='status-pill status-{sanitize_text(task['status'])}'>{sanitize_text(task['status'])}</span></div>"
            f"<div class='result-meta'><strong>Batch:</strong> {sanitize_text(task['batch_id'])}</div>"
            f"<div class='result-meta'><strong>Group:</strong> {sanitize_text(task['experiment_group'] or '-')}</div>"
            f"<div class='result-prompt'>{sanitize_text(task['prompt'])}</div>"
            f"<audio controls preload='none' src='{sanitize_text(src)}'></audio>"
            f"<div class='result-meta'><strong>File:</strong> {sanitize_text(task['audio_path'])}</div>"
            "</div>"
        )
    return "<div class='results-grid'>" + "".join(cards) + "</div>"


def build_status_markdown(summary: dict[str, Any]) -> str:
    """Render batch counters for the experiment dashboard."""
    if not summary["batch_id"]:
        return "No batch selected. Submit prompts to create an experiment batch."
    return (
        f"**Batch:** `{summary['batch_id']}`  \n"
        f"**Workers:** {summary['max_workers']}  \n"
        f"**Total:** {summary['total']} | **Queued:** {summary['queued']} | **Running:** {summary['running']} | "
        f"**Completed:** {summary['completed']} | **Failed:** {summary['failed']} | **Cancelled:** {summary['cancelled']}"
    )


def update_batch_dashboard(batch_id: str | None, status_filter: str) -> tuple[Any, ...]:
    """Refresh batch selector, queue table, status panel, results, and export file."""
    manager = get_batch_manager()
    choices = manager.batch_choices()
    selected = batch_id if batch_id in choices else (choices[0] if choices else None)
    tasks = manager.tasks_for_batch(selected, status_filter=status_filter)
    summary = manager.batch_summary(selected)
    export_path = manager.export_batch(selected)
    status_text = build_status_markdown(summary)
    return (
        gr.update(choices=choices, value=selected),
        gr.update(value=build_task_rows(tasks)),
        status_text,
        build_results_html(manager.tasks_for_batch(selected)),
        export_path,
        summary,
    )


def submit_sao_batch(
    prompt_batch_text: str,
    experiment_group: str,
    note: str,
    seconds: int,
    seed: int,
    steps: int,
    cfg_scale: float,
    sigma_min: float,
    sigma_max: float,
    sampler_type: str,
    save_history: bool,
):
    """Create a new SAO experiment batch and enqueue one task per prompt."""
    prompts = parse_prompt_lines(prompt_batch_text)
    batch_id = get_batch_manager().submit_batch(
        prompts=prompts,
        seconds=seconds,
        seed=seed,
        steps=steps,
        cfg_scale=cfg_scale,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sampler_type=sampler_type,
        experiment_group=experiment_group.strip(),
        note=note.strip(),
        save_history=save_history,
    )
    prompt_table = [[str(index), prompt, experiment_group.strip(), note.strip()] for index, prompt in enumerate(prompts, start=1)]
    dashboard = update_batch_dashboard(batch_id, "all")
    return (batch_id, prompt_table, *dashboard)


def cancel_waiting_tasks(batch_id: str | None, status_filter: str):
    """Cancel queued tasks in the selected batch."""
    if batch_id:
        get_batch_manager().cancel_waiting(batch_id)
    return update_batch_dashboard(batch_id, status_filter)


def retry_failed_tasks(batch_id: str | None, status_filter: str):
    """Re-enqueue failed tasks in the selected batch."""
    if batch_id:
        get_batch_manager().retry_failed(batch_id)
    return update_batch_dashboard(batch_id, status_filter)


def clear_completed_tasks(batch_id: str | None, status_filter: str):
    """Remove completed tasks from the active dashboard while keeping files on disk."""
    if batch_id:
        get_batch_manager().clear_completed(batch_id)
    return update_batch_dashboard(batch_id, status_filter)


def export_selected_batch(batch_id: str | None, status_filter: str):
    """Refresh export artifact for the selected batch."""
    return update_batch_dashboard(batch_id, status_filter)


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


def sao_workbench_css() -> str:
    """Return lightweight styles for the experiment-oriented SAO workbench."""
    return """
    .result-empty {
        padding: 16px;
        border: 1px dashed #999;
        border-radius: 12px;
        color: #666;
        background: rgba(0, 0, 0, 0.03);
    }
    .results-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 12px;
    }
    .result-card {
        border: 1px solid #ddd;
        border-radius: 12px;
        padding: 12px;
        background: white;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .result-card audio {
        width: 100%;
    }
    .result-card-header {
        display: flex;
        justify-content: space-between;
        gap: 8px;
        align-items: center;
    }
    .result-prompt {
        font-size: 0.92rem;
        line-height: 1.45;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .result-meta {
        color: #555;
        font-size: 0.85rem;
        word-break: break-all;
    }
    .status-pill {
        font-size: 0.78rem;
        padding: 2px 8px;
        border-radius: 999px;
        background: #e5e7eb;
    }
    .status-completed { background: #dcfce7; }
    .status-running { background: #dbeafe; }
    .status-failed { background: #fee2e2; }
    .status-queued { background: #fef3c7; }
    .status-cancelled { background: #e5e7eb; }
    """


def build_app() -> gr.Blocks:
    """Build and return the unified multi-page Gradio app."""
    ensure_dirs()

    with gr.Blocks(title="Qwen + SAO Experiment Lab", css=sao_workbench_css()) as demo:
        gr.Markdown(
            """
            # Qwen + SAO Experiment Lab
            One service with two pages:
            1) SAO-only quick generation + multi-task experiment workbench
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

            with gr.Tab("Experiment Workbench"):
                gr.Markdown(
                    """
                    ### Multi-prompt SAO experiment workbench
                    - 每行一个 prompt，提交后会拆成独立任务。
                    - 队列表格会持续刷新，已完成任务会立即出现在下方结果区。
                    - 导出文件会保存本批次 prompt、状态、时间、输出路径与错误信息。
                    """
                )
                batch_state = gr.State(value=None)
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt_batch_text = gr.Textbox(
                            label="Batch Prompts (one prompt per line)",
                            lines=10,
                            value="\n".join([
                                _DEFAULT_SAO_PROMPT,
                                "Format: Solo | Genre: Ambient | Sub-genre: Texture | Instruments: warm pad, airy noise | Moods: calm, floating | Styles: cinematic, minimal | Tempo: Slow | BPM: 70 | Details: evolving airy motion | Quality: high-quality, stereo",
                            ]),
                        )
                        experiment_group = gr.Textbox(label="Experiment Group", placeholder="例如 baseline / prompt-v2")
                        experiment_note = gr.Textbox(label="Batch Note", lines=3, placeholder="记录本批次的实验目的、控制变量或备注")
                        prompt_table_preview = gr.Dataframe(
                            headers=["row", "prompt", "group", "note"],
                            value=experiment_table_value(
                                "\n".join([
                                    _DEFAULT_SAO_PROMPT,
                                    "Format: Solo | Genre: Ambient | Sub-genre: Texture | Instruments: warm pad, airy noise | Moods: calm, floating | Styles: cinematic, minimal | Tempo: Slow | BPM: 70 | Details: evolving airy motion | Quality: high-quality, stereo",
                                ]),
                                "",
                                "",
                            ),
                            interactive=False,
                            wrap=True,
                            label="Prompt Preview",
                        )
                        submit_batch_button = gr.Button("Submit Batch", variant="primary")

                    with gr.Column(scale=1):
                        batch_status = gr.Markdown(value="No batch selected. Submit prompts to create an experiment batch.")
                        batch_selector = gr.Dropdown(label="Experiment Batch", choices=[])
                        status_filter = gr.Radio(label="Status Filter", choices=_TASK_FILTER_OPTIONS, value="all")
                        export_file = gr.File(label="Batch Metadata Export (JSON)")
                        batch_summary_json = gr.JSON(label="Batch Summary")
                        with gr.Row():
                            cancel_waiting_button = gr.Button("Cancel Waiting")
                            retry_failed_button = gr.Button("Retry Failed")
                            clear_completed_button = gr.Button("Clear Completed")
                        export_button = gr.Button("Refresh Export")

                queue_table = gr.Dataframe(
                    headers=[
                        "task_id",
                        "prompt",
                        "status",
                        "submitted_at",
                        "started_at",
                        "ended_at",
                        "elapsed",
                        "audio_path",
                        "error",
                        "group",
                        "note",
                    ],
                    value=[],
                    interactive=False,
                    wrap=True,
                    label="Task Queue",
                )
                results_html = gr.HTML(value=build_results_html([]), label="Completed Results")
                refresh_timer = gr.Timer(value=2.0)

                prompt_batch_text.change(
                    fn=experiment_table_value,
                    inputs=[prompt_batch_text, experiment_group, experiment_note],
                    outputs=prompt_table_preview,
                )
                experiment_group.change(
                    fn=experiment_table_value,
                    inputs=[prompt_batch_text, experiment_group, experiment_note],
                    outputs=prompt_table_preview,
                )
                experiment_note.change(
                    fn=experiment_table_value,
                    inputs=[prompt_batch_text, experiment_group, experiment_note],
                    outputs=prompt_table_preview,
                )

                submit_batch_button.click(
                    fn=submit_sao_batch,
                    inputs=[
                        prompt_batch_text,
                        experiment_group,
                        experiment_note,
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
                        batch_state,
                        prompt_table_preview,
                        batch_selector,
                        queue_table,
                        batch_status,
                        results_html,
                        export_file,
                        batch_summary_json,
                    ],
                )

                status_filter.change(
                    fn=update_batch_dashboard,
                    inputs=[batch_selector, status_filter],
                    outputs=[batch_selector, queue_table, batch_status, results_html, export_file, batch_summary_json],
                )
                batch_selector.change(
                    fn=update_batch_dashboard,
                    inputs=[batch_selector, status_filter],
                    outputs=[batch_selector, queue_table, batch_status, results_html, export_file, batch_summary_json],
                )
                refresh_timer.tick(
                    fn=update_batch_dashboard,
                    inputs=[batch_selector, status_filter],
                    outputs=[batch_selector, queue_table, batch_status, results_html, export_file, batch_summary_json],
                )
                cancel_waiting_button.click(
                    fn=cancel_waiting_tasks,
                    inputs=[batch_selector, status_filter],
                    outputs=[batch_selector, queue_table, batch_status, results_html, export_file, batch_summary_json],
                )
                retry_failed_button.click(
                    fn=retry_failed_tasks,
                    inputs=[batch_selector, status_filter],
                    outputs=[batch_selector, queue_table, batch_status, results_html, export_file, batch_summary_json],
                )
                clear_completed_button.click(
                    fn=clear_completed_tasks,
                    inputs=[batch_selector, status_filter],
                    outputs=[batch_selector, queue_table, batch_status, results_html, export_file, batch_summary_json],
                )
                export_button.click(
                    fn=export_selected_batch,
                    inputs=[batch_selector, status_filter],
                    outputs=[batch_selector, queue_table, batch_status, results_html, export_file, batch_summary_json],
                )

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
    app.queue(default_concurrency_limit=8)
    app.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", os.environ.get("GRADIO_EXPERIMENT_PORT", "7860"))),
        allowed_paths=[str(EXPERIMENT_DIR)],
    )

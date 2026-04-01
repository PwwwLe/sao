"""FastAPI service that compiles raw prompts into structured SAO prompt JSON."""

from __future__ import annotations

import getpass
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# =========================
# 1. Load model
# =========================

MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen2-Audio-7B-Instruct")
LOCAL_MODEL_DIR = os.environ.get("QWEN_LOCAL_MODEL_DIR", "").strip()


def _infer_qwen_local_model_dir(model_id: str) -> str:
    """
    Try to infer a local HuggingFace Hub cache directory for the given model_id.
    """
    if "/" not in model_id:
        return ""

    org, name = model_id.split("/", 1)
    model_cache_leaf = f"models--{org}--{name}"

    cache_roots: list[Path] = []

    hf_hub_cache = os.environ.get("HF_HUB_CACHE", "").strip()
    if hf_hub_cache:
        cache_roots.append(Path(hf_hub_cache))

    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        cache_roots.append(Path(hf_home) / "hub")

    cache_roots.append(Path(os.path.expanduser("~/.cache/huggingface/hub")))

    try:
        username = getpass.getuser()
        cache_roots.append(Path(f"/data01/audio_group/{username}/.cache/huggingface/hub"))
    except Exception:
        pass

    seen: set[str] = set()
    uniq_cache_roots: list[Path] = []
    for root in cache_roots:
        try:
            key = str(root.resolve())
        except Exception:
            key = str(root)
        if key in seen:
            continue
        seen.add(key)
        uniq_cache_roots.append(root)

    for hub_cache_root in uniq_cache_roots:
        model_cache_base = hub_cache_root / model_cache_leaf
        if not model_cache_base.is_dir():
            continue

        snapshots_dir = model_cache_base / "snapshots"
        if snapshots_dir.is_dir():
            candidates = [p for p in snapshots_dir.iterdir() if p.is_dir()]
            if candidates:
                candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return str(candidates[0])

        return str(model_cache_base)

    return ""


if not LOCAL_MODEL_DIR:
    LOCAL_MODEL_DIR = _infer_qwen_local_model_dir(MODEL_ID).strip()

USE_LOCAL_MODEL = bool(LOCAL_MODEL_DIR)

if USE_LOCAL_MODEL and not os.path.isdir(LOCAL_MODEL_DIR):
    raise RuntimeError(f"QWEN_LOCAL_MODEL_DIR does not exist: {LOCAL_MODEL_DIR}")

MODEL_SOURCE = LOCAL_MODEL_DIR if USE_LOCAL_MODEL else MODEL_ID
print(f"[QWEN] Loading model from: {MODEL_SOURCE}")
print(f"[QWEN] local_files_only={USE_LOCAL_MODEL}")

processor = AutoProcessor.from_pretrained(
    MODEL_SOURCE,
    local_files_only=USE_LOCAL_MODEL,
)
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    MODEL_SOURCE,
    dtype="auto",
    device_map="auto",
    local_files_only=USE_LOCAL_MODEL,
)
model.eval()

# =========================
# 2. Prompt schema and system prompt
# =========================

SCHEMA_PROMPT_SCHEMA = """
{
  "audio_type": "AMBIENCE",

  "environment": "",
  "time_of_day": "",
  "weather": "",

  "sound_sources_primary": "",
  "sound_sources_secondary": "",
  "background_textures": "",

  "spatial_characteristics": "",
  "distance_profile": "",
  "stereo_width": "",

  "temporal_evolution": "",
  "event_density": "",

  "acoustic_properties": "",

  "mood": "",
  "immersion_level": "",
  "recording_style": "",

  "looping": "",
  "duration_hint": "",
  "negative_prompt": "",

  "sao_prompt": ""
}
"""

COT_PROMPT_SCHEMA = """
{
  "audio_type": "AMBIENCE",

  "environment": "",
  "time_of_day": "",
  "weather": "",

  "sound_sources_primary": "",
  "sound_sources_secondary": "",
  "background_textures": "",

  "spatial_characteristics": "",
  "distance_profile": "",
  "stereo_width": "",

  "temporal_evolution": "",
  "event_density": "",

  "acoustic_properties": "",

  "mood": "",
  "immersion_level": "",
  "recording_style": "",

  "looping": "",
  "duration_hint": "",
  "negative_prompt": "",

  "cot_trace": {
    "scene_family": "",
    "bed_layer": "",
    "spot_layer": "",
    "dynamic_layer": "",
    "foreground_background_tendency": "",
    "stationarity_variability": "",
    "loopability_risk": "",
    "control_decisions": []
  },

  "reasoning_tags": [],
  "sao_prompt": ""
}
"""

AMBIENCE_SCHEMA_SYSTEM_PROMPT_TEMPLATE = """
You are a professional Stable Audio Open (SAO) prompt engineer specialized in AMBIENCE and soundscape synthesis.

CRITICAL REQUIREMENTS:
- Output ONLY valid JSON.
- ALL text values MUST be in ENGLISH.
- Do NOT hallucinate. Leave unknown fields empty.
- Focus on environmental audio, not music.
- Do NOT output explanations or reasoning.

GOAL:
Convert the user prompt into a structured AMBIENCE representation optimized for Stable Audio Open.

IMPORTANT:
Stable Audio Open is trained on metadata-like prompts formed by concatenating natural language descriptions and tags.
Therefore:
- Use concise descriptive phrases
- Prefer metadata-style wording
- Prefer noun phrases over full sentences
- Avoid unnecessary connectors
- Keep the final sao_prompt compact and controllable

FIELD GUIDELINES:

[environment]
Type of environment (forest, city street, cave, office, ocean, subway station, etc.)

[time_of_day]
morning / afternoon / dusk / night / sunset

[weather]
rain / wind / storm / fog / clear

[sound_sources_primary]
Dominant sound sources, max 3 items

[sound_sources_secondary]
Intermittent or supporting sources

[background_textures]
Continuous ambience bed or low-level textures

[spatial_characteristics]
open / enclosed / wide / narrow / reflective

[distance_profile]
close / mid / far / layered

[stereo_width]
narrow / moderate / wide

[temporal_evolution]
static / slow evolving / dynamic / intermittent

[event_density]
sparse / moderate / dense

[acoustic_properties]
reverb, reflections, spectral feel, masking

[mood]
calm / eerie / tense / peaceful / neutral / immersive

[immersion_level]
background / immersive / cinematic

[recording_style]
field recording / designed soundscape / cinematic mix

[looping]
Use "loopable, seamless loop" only if explicitly requested or clearly implied

[duration_hint]
Copy only if provided

SAO PROMPT CONSTRUCTION:
Construct sao_prompt as ONE LINE using compact metadata-style phrasing.

Rules:
- Merge sounds in this order: sound_sources_primary; sound_sources_secondary; background_textures
- Keep sao_prompt concise
- Avoid long sentences
- Leave fields empty if unknown
- Output valid JSON only

OUTPUT JSON SCHEMA:
{prompt_schema}
"""

AMBIENCE_COT_SYSTEM_PROMPT_TEMPLATE = """
You are a professional Stable Audio Open (SAO) prompt engineer specialized in AMBIENCE and soundscape synthesis.

CRITICAL REQUIREMENTS:
- Output ONLY valid JSON.
- ALL text values MUST be in ENGLISH.
- Do NOT hallucinate. Leave unknown fields empty.
- Focus on environmental audio, not music.
- Do NOT output free-form explanations.
- You MUST output an explicit cot_trace object.
- Represent reasoning as short structured fields and short tags, not long prose.

GOAL:
Convert the user prompt into a structured AMBIENCE representation optimized for Stable Audio Open.

IMPORTANT:
Stable Audio Open is trained on metadata-like prompts formed by concatenating natural language descriptions and tags.
Therefore:
- Use concise descriptive phrases
- Prefer metadata-style wording
- Prefer noun phrases over full sentences
- Avoid unnecessary connectors
- Keep the final sao_prompt compact and controllable

EXPLICIT REASONING PROTOCOL:
You MUST reason in this structured way and expose the result in cot_trace and reasoning_tags:

1. Classify the scene family:
   natural outdoors / weather and water / urban outdoors / transportation and transit /
   indoor built environments / industrial and infrastructure / enclosed or subterranean /
   synthetic or abstract

2. Decompose the ambience into:
   - bed_layer: continuous ambience bed
   - spot_layer: sparse salient sound events
   - dynamic_layer: slow changes, movement, or modulation

3. Infer control targets:
   - foreground_background_tendency
   - stationarity_variability
   - loopability_risk

4. Write short control_decisions as imperative constraints for prompt compilation, e.g.:
   - "avoid close thunder cracks"
   - "keep birds distant and sparse"
   - "use wide diffuse stereo"
   - "maintain background dominance"

5. Write short reasoning_tags that are metadata-like and suitable for SAO conditioning.
   Examples:
   - "background-dominant"
   - "wide diffuse stereo"
   - "high stationarity"
   - "sparse spot events"
   - "loop-friendly rain bed"

FIELD GUIDELINES:

[environment]
Type of environment (forest, city street, cave, office, ocean, subway station, etc.)

[time_of_day]
morning / afternoon / dusk / night / sunset

[weather]
rain / wind / storm / fog / clear

[sound_sources_primary]
Dominant sound sources, max 3 items

[sound_sources_secondary]
Intermittent or supporting sources

[background_textures]
Continuous ambience bed or low-level textures

[spatial_characteristics]
open / enclosed / wide / narrow / reflective

[distance_profile]
close / mid / far / layered

[stereo_width]
narrow / moderate / wide

[temporal_evolution]
static / slow evolving / dynamic / intermittent

[event_density]
sparse / moderate / dense

[acoustic_properties]
reverb, reflections, spectral feel, masking

[mood]
calm / eerie / tense / peaceful / neutral / immersive

[immersion_level]
background / immersive / cinematic

[recording_style]
field recording / designed soundscape / cinematic mix

[looping]
Use "loopable, seamless loop" only if explicitly requested or clearly implied

[duration_hint]
Copy only if provided

OUTPUT JSON SCHEMA:
{prompt_schema}
"""

# =========================
# 2.5 Schema normalization
# =========================

SCHEMA_DEFAULTS: dict[str, Any] = {
    "audio_type": "AMBIENCE",
    "environment": "",
    "time_of_day": "",
    "weather": "",
    "sound_sources_primary": "",
    "sound_sources_secondary": "",
    "background_textures": "",
    "spatial_characteristics": "",
    "distance_profile": "",
    "stereo_width": "",
    "temporal_evolution": "",
    "event_density": "",
    "acoustic_properties": "",
    "mood": "",
    "immersion_level": "",
    "recording_style": "",
    "looping": "",
    "duration_hint": "",
    "negative_prompt": "",
    "sao_prompt": "",
}

COT_TRACE_DEFAULTS: dict[str, Any] = {
    "scene_family": "",
    "bed_layer": "",
    "spot_layer": "",
    "dynamic_layer": "",
    "foreground_background_tendency": "",
    "stationarity_variability": "",
    "loopability_risk": "",
    "control_decisions": [],
}

COT_DEFAULTS: dict[str, Any] = {
    **SCHEMA_DEFAULTS,
    "cot_trace": COT_TRACE_DEFAULTS.copy(),
    "reasoning_tags": [],
}

KEY_ALIASES = {
    "sound_sources-secondary": "sound_sources_secondary",
    "sound_sources.secondary": "sound_sources_secondary",
    "sound_sources secondary": "sound_sources_secondary",
    "sound_sources-primary": "sound_sources_primary",
    "sound_sources.primary": "sound_sources_primary",
    "sound_sources primary": "sound_sources_primary",
    "time-of-day": "time_of_day",
    "distance-profile": "distance_profile",
    "stereo-width": "stereo_width",
    "event-density": "event_density",
    "recording-style": "recording_style",
    "duration-hint": "duration_hint",
    "negative-prompt": "negative_prompt",
    "scene-family": "scene_family",
    "bed-layer": "bed_layer",
    "spot-layer": "spot_layer",
    "dynamic-layer": "dynamic_layer",
    "foreground-background-tendency": "foreground_background_tendency",
    "stationarity-variability": "stationarity_variability",
    "loopability-risk": "loopability_risk",
    "control-decisions": "control_decisions",
    "reasoning-tags": "reasoning_tags",
}


def _clean_text(s: str) -> str:
    s = str(s).replace("\r", "\n")
    s = re.sub(r"\s+", " ", s.strip())
    return s


def _contains_code_fence(text: str) -> bool:
    return "```" in text


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    fenced = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text


def _extract_first_json_object(text: str) -> str:
    """
    Extract the first balanced JSON object from possibly noisy model output.
    """
    text = _strip_markdown_fences(text)

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start '{' found in model output")

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(text)):
        ch = text[idx]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]

    raise ValueError("No balanced JSON object found in model output")


def _loads_model_json(text: str) -> dict[str, Any]:
    candidate = _extract_first_json_object(text)
    obj = json.loads(candidate)
    if not isinstance(obj, dict):
        raise ValueError("Model output JSON root must be an object")
    return obj


def _normalize_key(key: str) -> str:
    key = str(key).strip()
    if key in KEY_ALIASES:
        return KEY_ALIASES[key]
    key = key.replace("-", "_").replace(".", "_").replace(" ", "_")
    return key


def _normalize_string_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        parts = [_clean_text(x) for x in value if _clean_text(x)]
        return "; ".join(parts)
    if isinstance(value, dict):
        return _clean_text(json.dumps(value, ensure_ascii=False))
    return _clean_text(str(value))


def _normalize_short_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_clean_text(x) for x in value if _clean_text(x)]
    if value is None:
        return []
    text = _normalize_string_field(value)
    if not text:
        return []
    return [text]


def _normalize_dict_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {_normalize_key(k): _normalize_dict_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_dict_keys(v) for v in obj]
    return obj


def _normalize_payload(obj: dict[str, Any], mode: str) -> dict[str, Any]:
    """
    Normalize parsed model JSON into canonical schema keys and types.
    """
    obj = _normalize_dict_keys(obj)

    defaults = COT_DEFAULTS if mode == "cot" else SCHEMA_DEFAULTS
    normalized: dict[str, Any] = {}

    for key, default_value in defaults.items():
        if key in ("cot_trace", "reasoning_tags", "sao_prompt"):
            continue

        raw = obj.get(key, default_value)
        normalized[key] = _normalize_string_field(raw)

    if mode == "cot":
        raw_cot = obj.get("cot_trace", {})
        if not isinstance(raw_cot, dict):
            raw_cot = {}

        raw_cot = _normalize_dict_keys(raw_cot)
        cot_trace: dict[str, Any] = {}

        for key, default_value in COT_TRACE_DEFAULTS.items():
            raw = raw_cot.get(key, default_value)
            if key == "control_decisions":
                cot_trace[key] = _normalize_short_string_list(raw)
            else:
                cot_trace[key] = _normalize_string_field(raw)

        normalized["cot_trace"] = cot_trace
        normalized["reasoning_tags"] = _normalize_short_string_list(obj.get("reasoning_tags", []))

    return normalized


def _join_nonempty(parts: list[str]) -> str:
    return "; ".join([_clean_text(p) for p in parts if _clean_text(p)])


def _truncate_to_token_range(text: str, min_tokens: int = 20, max_tokens: int = 80) -> str:
    words = text.split()
    if len(words) <= max_tokens:
        return " ".join(words)
    return " ".join(words[:max_tokens])


def rebuild_sao_prompt(obj: dict[str, Any], mode: str, min_tokens: int = 20, max_tokens: int = 80) -> str:
    sounds = _join_nonempty([
        obj.get("sound_sources_primary", ""),
        obj.get("sound_sources_secondary", ""),
        obj.get("background_textures", ""),
    ])

    spatial = _join_nonempty([
        obj.get("spatial_characteristics", ""),
        obj.get("distance_profile", ""),
        obj.get("stereo_width", ""),
    ])

    temporal = _join_nonempty([
        obj.get("temporal_evolution", ""),
        obj.get("event_density", ""),
    ])

    details_parts = [
        obj.get("looping", ""),
        obj.get("duration_hint", ""),
    ]

    if mode == "cot":
        cot = obj.get("cot_trace", {}) if isinstance(obj.get("cot_trace", {}), dict) else {}
        reasoning_tags = _normalize_short_string_list(obj.get("reasoning_tags", []))
        control_decisions = _normalize_short_string_list(cot.get("control_decisions", []))

        cot_controls = _join_nonempty([
            cot.get("scene_family", ""),
            cot.get("bed_layer", ""),
            cot.get("spot_layer", ""),
            cot.get("dynamic_layer", ""),
            cot.get("foreground_background_tendency", ""),
            cot.get("stationarity_variability", ""),
            cot.get("loopability_risk", ""),
        ])

        details_parts.extend([
            cot_controls,
            "; ".join(reasoning_tags),
            "; ".join(control_decisions),
        ])

    details = _join_nonempty(details_parts)

    prompt = (
        f"Environment: {obj.get('environment','')} | "
        f"Time: {obj.get('time_of_day','')} | "
        f"Weather: {obj.get('weather','')} | "
        f"Sounds: {sounds} | "
        f"Spatial: {spatial} | "
        f"Temporal: {temporal} | "
        f"Acoustics: {obj.get('acoustic_properties','')} | "
        f"Mood: {obj.get('mood','')} | "
        f"Style: {obj.get('recording_style','')} | "
        f"Details: {details} | "
        f"Quality: high-quality, stereo"
    )

    prompt = _clean_text(prompt)
    prompt = _truncate_to_token_range(prompt, min_tokens=min_tokens, max_tokens=max_tokens)
    return prompt


def postprocess_structured_output(raw_output: str, mode: str, min_tokens: int = 20, max_tokens: int = 80) -> str:
    """
    Strictly parse, normalize, and rebuild the final SAO prompt.

    This function must never preserve raw noisy model output inside sao_prompt.
    """
    parsed = _loads_model_json(raw_output)
    normalized = _normalize_payload(parsed, mode=mode)
    normalized["sao_prompt"] = rebuild_sao_prompt(
        normalized,
        mode=mode,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )

    if not normalized["sao_prompt"]:
        raise ValueError("Post-processed sao_prompt is empty")

    return json.dumps(normalized, ensure_ascii=False)


def safe_postprocess_structured_output(raw_output: str, mode: str, min_tokens: int = 20, max_tokens: int = 80) -> str:
    """
    Repair-oriented wrapper.

    Unlike the old implementation, this function does NOT silently inject raw output
    into sao_prompt. If output cannot be repaired into valid schema JSON, it fails fast.
    """
    return postprocess_structured_output(
        raw_output,
        mode=mode,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )


def _get_mode_default_schema(mode: str) -> str:
    return COT_PROMPT_SCHEMA if mode == "cot" else SCHEMA_PROMPT_SCHEMA


# =========================
# 3. FastAPI initialization
# =========================

app = FastAPI(
    title="Qwen2-Audio Prompt Compiler",
    version="2.1"
)

# =========================
# 4. Request/response models
# =========================

class PromptRequest(BaseModel):
    raw_prompt: str
    max_new_tokens: int = 768
    system_prompt: str | None = None
    prompt_schema: str | None = None
    min_sao_tokens: int = 20
    max_sao_tokens: int = 80


class PromptResponse(BaseModel):
    structured_prompt: str


class PromptDebugResponse(BaseModel):
    structured_prompt: str
    parsed_json: dict | None = None


# =========================
# 5. Core inference functions
# =========================

def build_system_prompt(
    mode: str = "schema",
    system_prompt: str | None = None,
    prompt_schema: str | None = None
) -> str:
    schema = prompt_schema or _get_mode_default_schema(mode)

    if system_prompt:
        return system_prompt

    if mode == "cot":
        return AMBIENCE_COT_SYSTEM_PROMPT_TEMPLATE.format(prompt_schema=schema)

    return AMBIENCE_SCHEMA_SYSTEM_PROMPT_TEMPLATE.format(prompt_schema=schema)


def refine_prompt(
    raw_prompt: str,
    mode: str = "schema",
    max_new_tokens: int = 768,
    system_prompt: str | None = None,
    prompt_schema: str | None = None,
    min_sao_tokens: int = 20,
    max_sao_tokens: int = 80
) -> str:
    resolved_system_prompt = build_system_prompt(
        mode=mode,
        system_prompt=system_prompt,
        prompt_schema=prompt_schema
    )

    conversation = [
        {"role": "system", "content": resolved_system_prompt},
        {
            "role": "user",
            "content": f"""
IMPORTANT: Respond in ENGLISH ONLY.
Return valid JSON ONLY.
Do not use markdown.
Do not wrap JSON in code fences.

User prompt:
{raw_prompt}
"""
        },
    ]

    text = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(text=text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=True,
        )

    gen = output_ids[:, inputs["input_ids"].shape[1]:]
    out = processor.batch_decode(
        gen,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return safe_postprocess_structured_output(
        out,
        mode=mode,
        min_tokens=min_sao_tokens,
        max_tokens=max_sao_tokens
    )


# =========================
# 6. HTTP endpoints
# =========================

@app.post("/refine_prompt/schema", response_model=PromptResponse)
def refine_prompt_schema_api(req: PromptRequest):
    try:
        structured = refine_prompt(
            raw_prompt=req.raw_prompt,
            mode="schema",
            max_new_tokens=req.max_new_tokens,
            system_prompt=req.system_prompt,
            prompt_schema=req.prompt_schema,
            min_sao_tokens=req.min_sao_tokens,
            max_sao_tokens=req.max_sao_tokens
        )
        return {"structured_prompt": structured}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Schema prompt refinement failed: {exc}") from exc


@app.post("/refine_prompt/cot", response_model=PromptResponse)
def refine_prompt_cot_api(req: PromptRequest):
    try:
        structured = refine_prompt(
            raw_prompt=req.raw_prompt,
            mode="cot",
            max_new_tokens=req.max_new_tokens,
            system_prompt=req.system_prompt,
            prompt_schema=req.prompt_schema,
            min_sao_tokens=req.min_sao_tokens,
            max_sao_tokens=req.max_sao_tokens
        )
        return {"structured_prompt": structured}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"CoT prompt refinement failed: {exc}") from exc


@app.post("/refine_prompt/cot_debug", response_model=PromptDebugResponse)
def refine_prompt_cot_debug_api(req: PromptRequest):
    try:
        structured = refine_prompt(
            raw_prompt=req.raw_prompt,
            mode="cot",
            max_new_tokens=req.max_new_tokens,
            system_prompt=req.system_prompt,
            prompt_schema=req.prompt_schema,
            min_sao_tokens=req.min_sao_tokens,
            max_sao_tokens=req.max_sao_tokens
        )

        parsed = json.loads(structured)
        return {
            "structured_prompt": structured,
            "parsed_json": parsed
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"CoT debug refinement failed: {exc}") from exc
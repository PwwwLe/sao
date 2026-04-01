"""Prompt transformation utilities for Qwen2Audio -> Stable Audio Open."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

# ------------------------------------------------------------
# Debug Logging Configuration
# ------------------------------------------------------------
LOG_PATH = "/data01/audio_group/d26_pengwenle/.cursor/debug-6190b4.log"
SESSION_ID = "6190b4"


def _debug_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict | None = None,
    run_id: str = "ambience_prompt_experiments",
) -> None:
    payload = {
        "sessionId": SESSION_ID,
        "id": f"log_{int(time.time() * 1000)}_{os.getpid()}",
        "timestamp": int(time.time() * 1000),
        "location": location,
        "message": message,
        "data": data or {},
        "runId": run_id,
        "hypothesisId": hypothesis_id,
    }
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


@dataclass(frozen=True)
class PromptTransform:
    raw_prompt: str
    reasoning: str | None
    reasoning_tags: str | None
    final_prompt: str
    structured_json: dict[str, Any] | None


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

SCHEMA_KEYS = {
    "audio_type",
    "environment",
    "time_of_day",
    "weather",
    "sound_sources_primary",
    "sound_sources_secondary",
    "background_textures",
    "spatial_characteristics",
    "distance_profile",
    "stereo_width",
    "temporal_evolution",
    "event_density",
    "acoustic_properties",
    "mood",
    "immersion_level",
    "recording_style",
    "looping",
    "duration_hint",
    "negative_prompt",
    "sao_prompt",
}

COT_TRACE_KEYS = {
    "scene_family",
    "bed_layer",
    "spot_layer",
    "dynamic_layer",
    "foreground_background_tendency",
    "stationarity_variability",
    "loopability_risk",
    "control_decisions",
}


def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())


def _contains_code_fence(text: str) -> bool:
    return "```" in text


def _looks_like_json_payload(text: str) -> bool:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return True
    if '"sao_prompt"' in text or '"audio_type"' in text or '"cot_trace"' in text:
        return True
    return False


def _normalize_key(key: str) -> str:
    key = str(key).strip()
    if key in KEY_ALIASES:
        return KEY_ALIASES[key]
    return key.replace("-", "_").replace(".", "_").replace(" ", "_")


def _normalize_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {_normalize_key(k): _normalize_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_keys(v) for v in obj]
    return obj


def _get_qwen_base_url() -> str:
    return os.environ.get("QWEN_SERVICE_URL_BASE", "http://127.0.0.1:8008").rstrip("/")


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()


def _parse_structured_prompt_text(structured_prompt_text: str) -> dict[str, Any]:
    """
    Strictly parse cleaned JSON returned by qwen2audio_server.py.
    """
    if not structured_prompt_text or not structured_prompt_text.strip():
        raise ValueError("Empty Qwen structured_prompt")

    if _contains_code_fence(structured_prompt_text):
        raise ValueError("Qwen structured_prompt still contains markdown code fences")

    data = json.loads(structured_prompt_text)
    if not isinstance(data, dict):
        raise ValueError("Qwen structured_prompt must decode to a JSON object")

    return _normalize_keys(data)


def _validate_sao_prompt(final_prompt: str) -> str:
    final_prompt = _clean_text(final_prompt)

    if not final_prompt:
        raise ValueError("Empty sao_prompt")

    if _contains_code_fence(final_prompt):
        raise ValueError("sao_prompt contains markdown code fences")

    if _looks_like_json_payload(final_prompt):
        raise ValueError("sao_prompt looks like raw JSON payload instead of final SAO prompt")

    return final_prompt


def _sanitize_structured_payload(data: dict[str, Any], condition: str) -> dict[str, Any]:
    """
    Second-layer defense on client side before writing cache.
    """
    data = _normalize_keys(data)

    sao_prompt = _validate_sao_prompt(str(data.get("sao_prompt", "")))
    data["sao_prompt"] = sao_prompt

    # Normalize top-level schema keys defensively
    clean: dict[str, Any] = {}

    for key in SCHEMA_KEYS:
        value = data.get(key)
        if key == "sao_prompt":
            clean[key] = sao_prompt
        else:
            if value is None:
                clean[key] = ""
            elif isinstance(value, list):
                clean[key] = "; ".join([_clean_text(x) for x in value if _clean_text(x)])
            elif isinstance(value, dict):
                clean[key] = _clean_text(json.dumps(value, ensure_ascii=False))
            else:
                clean[key] = _clean_text(value)

    if condition == "cot":
        raw_cot = data.get("cot_trace", {})
        if not isinstance(raw_cot, dict):
            raw_cot = {}
        raw_cot = _normalize_keys(raw_cot)

        cot_trace: dict[str, Any] = {}
        for key in COT_TRACE_KEYS:
            value = raw_cot.get(key)
            if key == "control_decisions":
                if isinstance(value, list):
                    cot_trace[key] = [_clean_text(x) for x in value if _clean_text(x)]
                elif value is None:
                    cot_trace[key] = []
                else:
                    cot_trace[key] = [_clean_text(value)]
            else:
                cot_trace[key] = "" if value is None else _clean_text(value)

        reasoning_tags = data.get("reasoning_tags", [])
        if isinstance(reasoning_tags, list):
            reasoning_tags = [_clean_text(x) for x in reasoning_tags if _clean_text(x)]
        elif reasoning_tags is None:
            reasoning_tags = []
        else:
            reasoning_tags = [_clean_text(reasoning_tags)]

        clean["cot_trace"] = cot_trace
        clean["reasoning_tags"] = reasoning_tags

    return clean


def _call_qwen_schema(raw_prompt: str) -> dict[str, Any]:
    base_url = _get_qwen_base_url()
    endpoint = f"{base_url}/refine_prompt/schema"

    payload = {
        "raw_prompt": raw_prompt,
        "max_new_tokens": int(os.environ.get("QWEN_MAX_NEW_TOKENS", "768")),
        "min_sao_tokens": int(os.environ.get("QWEN_MIN_SAO_TOKENS", "20")),
        "max_sao_tokens": int(os.environ.get("QWEN_MAX_SAO_TOKENS", "80")),
    }

    _debug_log(
        hypothesis_id="H2_qwen_schema_call",
        location="scripts/qwen_prompt.py:_call_qwen_schema",
        message="Calling Qwen schema endpoint",
        data={"endpoint": endpoint, "payload": payload},
    )

    data = _post_json(endpoint, payload)
    parsed = _parse_structured_prompt_text(data["structured_prompt"])
    return _sanitize_structured_payload(parsed, condition="structured")


def _call_qwen_cot(raw_prompt: str) -> dict[str, Any]:
    base_url = _get_qwen_base_url()
    endpoint = f"{base_url}/refine_prompt/cot"

    payload = {
        "raw_prompt": raw_prompt,
        "max_new_tokens": int(os.environ.get("QWEN_MAX_NEW_TOKENS", "768")),
        "min_sao_tokens": int(os.environ.get("QWEN_MIN_SAO_TOKENS", "20")),
        "max_sao_tokens": int(os.environ.get("QWEN_MAX_SAO_TOKENS", "80")),
    }

    _debug_log(
        hypothesis_id="H3_qwen_cot_call",
        location="scripts/qwen_prompt.py:_call_qwen_cot",
        message="Calling Qwen CoT endpoint",
        data={"endpoint": endpoint, "payload": payload},
    )

    data = _post_json(endpoint, payload)
    parsed = _parse_structured_prompt_text(data["structured_prompt"])
    return _sanitize_structured_payload(parsed, condition="cot")


def _stringify_reasoning(data: dict[str, Any]) -> tuple[str | None, str | None]:
    cot = data.get("cot_trace")
    reasoning_tags = data.get("reasoning_tags")

    reasoning_text = None
    reasoning_tags_text = None

    if isinstance(cot, dict):
        reasoning_text = json.dumps(cot, ensure_ascii=False)

    if isinstance(reasoning_tags, list):
        cleaned = [str(x).strip() for x in reasoning_tags if str(x).strip()]
        reasoning_tags_text = "; ".join(cleaned) if cleaned else None

    return reasoning_text, reasoning_tags_text


def build_transform(raw_prompt: str, condition: str) -> PromptTransform:
    if condition == "baseline":
        return PromptTransform(
            raw_prompt=raw_prompt,
            reasoning=None,
            reasoning_tags=None,
            final_prompt=raw_prompt,
            structured_json=None,
        )

    if condition == "structured":
        try:
            data = _call_qwen_schema(raw_prompt)
            final_prompt = _validate_sao_prompt(str(data.get("sao_prompt", "")))

            return PromptTransform(
                raw_prompt=raw_prompt,
                reasoning=None,
                reasoning_tags=None,
                final_prompt=final_prompt,
                structured_json=data,
            )
        except Exception as exc:
            print(f"[WARN] Qwen structured compile failed. error={exc}")
            raise

    if condition == "cot":
        try:
            data = _call_qwen_cot(raw_prompt)
            final_prompt = _validate_sao_prompt(str(data.get("sao_prompt", "")))
            reasoning_text, reasoning_tags_text = _stringify_reasoning(data)

            return PromptTransform(
                raw_prompt=raw_prompt,
                reasoning=reasoning_text,
                reasoning_tags=reasoning_tags_text,
                final_prompt=final_prompt,
                structured_json=data,
            )
        except Exception as exc:
            print(f"[WARN] Qwen CoT compile failed. error={exc}")
            raise

    raise ValueError(f"Unsupported condition: {condition}")
"""
Validate prompt_cache.json for:
1. truncation
2. markdown/code-fence pollution
3. inconsistent key naming
4. suspicious final_prompt payloads

Usage:
    python scripts/prompt_cache_validator.py --input artifacts/prompt_cache.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

CONDITIONS = ("baseline", "structured", "cot")

ALIAS_KEYS = {
    "sound_sources-secondary",
    "sound_sources.secondary",
    "sound_sources secondary",
    "sound_sources-primary",
    "sound_sources.primary",
    "sound_sources primary",
    "time-of-day",
    "distance-profile",
    "stereo-width",
    "event-density",
    "recording-style",
    "duration-hint",
    "negative-prompt",
    "scene-family",
    "bed-layer",
    "spot-layer",
    "dynamic-layer",
    "foreground-background-tendency",
    "stationarity-variability",
    "loopability-risk",
    "control-decisions",
    "reasoning-tags",
}


def _contains_code_fence(text: str) -> bool:
    return "```" in str(text)


def _looks_like_json_payload(text: str) -> bool:
    text = str(text).strip()
    if text.startswith("{") and text.endswith("}"):
        return True
    if '"audio_type"' in text or '"sao_prompt"' in text or '"cot_trace"' in text:
        return True
    return False


def _unbalanced_brackets(text: str) -> bool:
    pairs = {"{": "}", "[": "]"}
    stack: list[str] = []
    in_string = False
    escape = False

    for ch in str(text):
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
        elif ch in pairs:
            stack.append(ch)
        elif ch in ("}", "]"):
            if not stack:
                return True
            top = stack.pop()
            if pairs[top] != ch:
                return True

    return bool(stack) or in_string


def _find_alias_keys(obj: Any, path: str = "") -> list[str]:
    findings: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_path = f"{path}.{key}" if path else str(key)
            if key in ALIAS_KEYS:
                findings.append(key_path)
            findings.extend(_find_alias_keys(value, key_path))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            findings.extend(_find_alias_keys(value, f"{path}[{idx}]"))
    return findings


def _validate_entry(prompt_id: str, condition: str, entry: dict[str, Any]) -> list[str]:
    issues: list[str] = []

    final_prompt = entry.get("final_prompt")
    reasoning = entry.get("reasoning")
    structured_json = entry.get("structured_json")

    if final_prompt is None or not str(final_prompt).strip():
        issues.append(f"{prompt_id}.{condition}: final_prompt is empty")

    if final_prompt is not None:
        if _contains_code_fence(final_prompt):
            issues.append(f"{prompt_id}.{condition}: final_prompt contains markdown code fences")
        if _looks_like_json_payload(final_prompt):
            issues.append(f"{prompt_id}.{condition}: final_prompt looks like raw JSON instead of pure SAO prompt")
        if _unbalanced_brackets(final_prompt):
            issues.append(f"{prompt_id}.{condition}: final_prompt looks truncated or has unbalanced brackets")

    if reasoning is not None:
        if _contains_code_fence(reasoning):
            issues.append(f"{prompt_id}.{condition}: reasoning contains markdown code fences")
        if _unbalanced_brackets(reasoning):
            issues.append(f"{prompt_id}.{condition}: reasoning looks truncated or has unbalanced brackets")

    if structured_json is not None:
        alias_paths = _find_alias_keys(structured_json)
        for alias_path in alias_paths:
            issues.append(f"{prompt_id}.{condition}: inconsistent key alias found at {alias_path}")

        if isinstance(structured_json, dict):
            sao_prompt = structured_json.get("sao_prompt")
            if sao_prompt is None or not str(sao_prompt).strip():
                issues.append(f"{prompt_id}.{condition}: structured_json.sao_prompt is empty")
            else:
                if _contains_code_fence(sao_prompt):
                    issues.append(f"{prompt_id}.{condition}: structured_json.sao_prompt contains markdown code fences")
                if _looks_like_json_payload(sao_prompt):
                    issues.append(f"{prompt_id}.{condition}: structured_json.sao_prompt looks like raw JSON")
                if _unbalanced_brackets(sao_prompt):
                    issues.append(f"{prompt_id}.{condition}: structured_json.sao_prompt looks truncated")

            if condition == "cot":
                cot_trace = structured_json.get("cot_trace")
                if cot_trace is not None and not isinstance(cot_trace, dict):
                    issues.append(f"{prompt_id}.{condition}: cot_trace is not a dict")
                reasoning_tags = structured_json.get("reasoning_tags")
                if reasoning_tags is not None and not isinstance(reasoning_tags, list):
                    issues.append(f"{prompt_id}.{condition}: reasoning_tags is not a list")

    return issues


def validate_prompt_cache(cache: dict[str, Any]) -> list[str]:
    issues: list[str] = []

    if not isinstance(cache, dict):
        return ["Top-level prompt cache is not a JSON object"]

    for prompt_id, prompt_entry in cache.items():
        if not isinstance(prompt_entry, dict):
            issues.append(f"{prompt_id}: prompt entry is not an object")
            continue

        for condition in CONDITIONS:
            if condition not in prompt_entry:
                issues.append(f"{prompt_id}: missing condition '{condition}'")
                continue

            entry = prompt_entry[condition]
            if not isinstance(entry, dict):
                issues.append(f"{prompt_id}.{condition}: condition entry is not an object")
                continue

            issues.extend(_validate_entry(prompt_id, condition, entry))

    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to prompt_cache.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.input.open("r", encoding="utf-8") as handle:
        cache = json.load(handle)

    issues = validate_prompt_cache(cache)

    if issues:
        print("[Validator] INVALID prompt cache detected.")
        print(f"[Validator] Issue count: {len(issues)}")
        for issue in issues:
            print(f" - {issue}")
        sys.exit(1)

    print("[Validator] Prompt cache is valid.")


if __name__ == "__main__":
    main()
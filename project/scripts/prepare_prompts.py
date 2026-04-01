"""
Prepare and cache all prompt transformations before SAO generation.

This script is used to decouple:
1. Qwen prompt compilation stage
2. Stable Audio Open generation stage

Workflow:
- Load raw prompts
- For each enabled condition:
    - baseline: direct passthrough
    - structured / cot: call Qwen service once
- Save full PromptTransform-compatible cache to JSON

This allows Qwen to be shut down before the heavy SAO generation stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from qwen_prompt import build_transform

CONDITIONS = ("baseline", "structured", "cot")


class PromptPreparationRunner:
    """
    Prepare prompt transformations and persist them as a local cache.
    """

    def __init__(self, config_path: Path, output_path: Path) -> None:
        self.config_path = config_path.resolve()
        self.output_path = output_path.resolve()

        with self.config_path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)

        self.project_root = self.config_path.parent
        self.prompts = self._load_prompts(self.project_root / self.config["prompts_file"])

    @staticmethod
    def _load_prompts(prompts_path: Path) -> list[dict[str, Any]]:
        with prompts_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def iter_enabled_conditions(self) -> Iterable[str]:
        condition_config = self.config.get("conditions", {})
        for condition in CONDITIONS:
            if condition_config.get(condition, {}).get("enabled", True):
                yield condition

    def run(self) -> dict[str, dict[str, dict[str, Any]]]:
        """
        Returns cache of structure:
        {
          "prompt_id": {
            "baseline": {
              "raw_prompt": "...",
              "reasoning": null,
              "reasoning_tags": null,
              "final_prompt": "...",
              "structured_json": null
            },
            "structured": {...},
            "cot": {...}
          }
        }
        """
        cache: dict[str, dict[str, dict[str, Any]]] = {}

        enabled_conditions = list(self.iter_enabled_conditions())
        print(f"[Prepare] Enabled conditions: {enabled_conditions}")

        for prompt_record in self.prompts:
            prompt_id = prompt_record["id"]
            raw_prompt = prompt_record["prompt"]

            cache[prompt_id] = {}

            for condition in enabled_conditions:
                print(f"[Prepare] prompt_id={prompt_id} condition={condition}")

                transform = build_transform(raw_prompt, condition)

                cache[prompt_id][condition] = {
                    "raw_prompt": transform.raw_prompt,
                    "reasoning": transform.reasoning,
                    "reasoning_tags": transform.reasoning_tags,
                    "final_prompt": transform.final_prompt,
                    "structured_json": transform.structured_json,
                }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as handle:
            json.dump(cache, handle, ensure_ascii=False, indent=2)

        print(f"[Prepare] Prompt cache saved to: {self.output_path}")
        return cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "prompt_cache.json",
        help="Output path for prepared prompt cache JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = PromptPreparationRunner(args.config, args.output)
    runner.run()


if __name__ == "__main__":
    main()
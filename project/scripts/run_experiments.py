"""
Run all Stable Audio Open experiment conditions end-to-end.

This script orchestrates the full experimental pipeline:
1. Load configuration and prompt set
2. Apply prompt transformation strategies (baseline / structured / CoT)
3. Generate audio using Stable Audio Open (SAO)
4. Compute evaluation metrics
5. Aggregate and persist results

Designed for controlled prompt optimization experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import yaml

from evaluate import compute_audio_metrics, summarize_results, write_results
from generate_audio import GenerationConfig, StableAudioGenerator
from qwen_prompt import PromptTransform, build_transform

# ------------------------------------------------------------
# Experiment Conditions
# ------------------------------------------------------------
# These correspond to different prompt engineering strategies:
# - baseline: raw prompt
# - structured: structured prompt (e.g., JSON-like constraints)
# - cot: Chain-of-Thought enhanced prompt
CONDITIONS = ("baseline", "structured", "cot")

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
    """
    Lightweight structured logging for experiment tracing.

    Each log entry is JSONL formatted for downstream analysis.

    Args:
        hypothesis_id: Identifier for tracked hypothesis (for ablation/debugging)
        location: Code location where log is emitted
        message: Human-readable description
        data: Optional structured payload
        run_id: Logical experiment group identifier
    """
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
        # Logging must never interrupt experiment execution
        pass


class ExperimentRunner:
    """
    Core class responsible for running the full experiment pipeline.

    Responsibilities:
    - Load config and prompts
    - Initialize audio generator
    - Iterate over (prompt × condition × seed)
    - Generate audio + evaluate
    - Aggregate results
    """
    
    def __init__(self, config_path: Path, prompt_cache_path: Path | None = None) -> None:
        """
        Initialize experiment runner from YAML config.

        Args:
            config_path: Path to config.yaml
            prompt_cache_path: Path to precomputed prompt cache
        """
        self.config_path = config_path.resolve()
        
        # Load experiment configuration
        with self.config_path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
            
        self.project_root = self.config_path.parent
        
        # Load prompt dataset
        self.prompts = self._load_prompts(self.project_root / self.config["prompts_file"])
        
        # Output / metrics paths
        self.output_root = self.project_root / self.config["outputs_root"]
        self.metrics_file = self.project_root / self.config["metrics_file"]
        
        self.prompt_cache_path = prompt_cache_path.resolve() if prompt_cache_path else None
        self.prompt_cache = self._load_prompt_cache(self.prompt_cache_path)

         # Initialize generation config
        sampler = self.config["sampler"]
        
        self.generator = StableAudioGenerator(
            GenerationConfig(
                model_name=self.config["model_name"],
                sample_rate=int(self.config["sample_rate"]),
                duration_seconds=int(self.config["duration_seconds"]),
                device=self.config["device"],
                steps=int(sampler["steps"]),
                cfg_scale=float(sampler["cfg_scale"]),
                sigma_min=float(sampler["sigma_min"]),
                sigma_max=float(sampler["sigma_max"]),
                sampler_type=str(sampler["sampler_type"]),
            )
        )

    @staticmethod
    def _load_prompts(prompts_path: Path) -> list[dict]:
        """
        Load prompt definitions from JSON file.

        Expected format:
        [
            {"id": "...", "prompt": "..."},
            ...
        ]
        """
        with prompts_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
        
    @staticmethod
    def _load_prompt_cache(prompt_cache_path: Path | None) -> dict[str, dict[str, dict[str, Any]]]:
        if prompt_cache_path is None:
            return {}
        if not prompt_cache_path.is_file():
            raise FileNotFoundError(f"Prompt cache file not found: {prompt_cache_path}")
        with prompt_cache_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def iter_enabled_conditions(self) -> Iterable[str]:
        """
        Yield enabled experimental conditions based on config.

        Allows selectively disabling certain conditions without
        modifying the global CONDITIONS list.
        """
        condition_config = self.config.get("conditions", {})
        for condition in CONDITIONS:
            if condition_config.get(condition, {}).get("enabled", True):
                yield condition

    def build_seed(self, index: int) -> int:
        """
        Generate deterministic seed for reproducibility.

        Args:
            index: seed index (0..num_seeds-1)

        Returns:
            integer seed
        """
        return int(self.config["seed_offset"]) + index
    
    def _transform_from_cache(self, prompt_id: str, condition: str) -> PromptTransform:
        """
        Rebuild PromptTransform object from precomputed JSON cache.
        """
        try:
            cached = self.prompt_cache[prompt_id][condition]
        except KeyError as exc:
            raise KeyError(
                f"Missing prompt cache entry for prompt_id={prompt_id}, condition={condition}"
            ) from exc

        return PromptTransform(
            raw_prompt=cached["raw_prompt"],
            reasoning=cached.get("reasoning"),
            reasoning_tags=cached.get("reasoning_tags"),
            final_prompt=cached["final_prompt"],
            structured_json=cached.get("structured_json"),
        )

    def get_transform(self, prompt_id: str, raw_prompt: str, condition: str) -> PromptTransform:
        """
        Resolve final PromptTransform either from:
        1. local prompt cache (preferred)
        2. live Qwen transform call (fallback)
        """
        if self.prompt_cache:
            return self._transform_from_cache(prompt_id, condition)

        return build_transform(raw_prompt, condition)

    def run(self) -> list[dict]:
        """
        Execute the full experiment loop.

        Core loop:
            for prompt in prompts:
                for condition in enabled_conditions:
                    for seed in seeds:
                        → transform prompt
                        → generate audio
                        → compute metrics

        Returns:
            List of result rows (per-audio + summary)
        """
        enabled_conditions = list(self.iter_enabled_conditions())
        
        # Log experiment configuration
        _debug_log(
            hypothesis_id="H3_config_enabled_conditions",
            location="scripts/run_experiments.py:run(enabled_conditions)",
            message="Enabled conditions resolved from config.yaml",
            data={"enabled_conditions": enabled_conditions, "raw_conditions_section": self.config.get("conditions", {})},
        )

        rows: list[dict] = []
        
        # Track first-generation event per condition
        _logged_condition: set[str] = set()

        for prompt_record in self.prompts:
            prompt_id = prompt_record["id"]
            raw_prompt = prompt_record["prompt"]

            for condition in self.iter_enabled_conditions():
                
                # Apply prompt transformation strategy
                transform = build_transform(raw_prompt, condition)
                
                condition_dir = self.output_root / condition

                for seed_index in range(int(self.config["num_seeds"])):
                    seed = self.build_seed(seed_index)
                    filename = f"{condition}_{prompt_id}_{seed}.wav"
                    output_path = condition_dir / filename

                    print(f"prompt_id={prompt_id} condition={condition} seed={seed}")

                    # Log first generation per condition (used for debugging model load latency)
                    if condition not in _logged_condition and seed_index == 0:
                        _logged_condition.add(condition)
                        _debug_log(
                            hypothesis_id="H1_sao_model_load_blocking",
                            location="scripts/run_experiments.py:run(first_generate_call)",
                            message="About to generate audio for first seed in this condition",
                            data={"condition": condition, "prompt_id": prompt_id, "seed": seed, "output_path": str(output_path)},
                        )

                    # Log final prompt used for generation (critical for prompt research)
                    _debug_log(
                        hypothesis_id="H4_final_prompt_for_generation",
                        location="scripts/run_experiments.py:run(before_generate_to_file)",
                        message="Final prompt prepared for SAO generation",
                        data={
                            "condition": condition,
                            "prompt_id": prompt_id,
                            "seed": seed,
                            "raw_prompt": raw_prompt,
                            "final_prompt": transform.final_prompt,
                            "reasoning": transform.reasoning,
                            "reasoning_tags": transform.reasoning_tags,
                        },
                    )

                    # ------------------------------------------------------------
                    # Audio Generation
                    # ------------------------------------------------------------
                    self.generator.generate_to_file(
                        prompt=transform.final_prompt,
                        seed=seed,
                        output_path=output_path,
                    )

                    # ------------------------------------------------------------
                    # Metric Computation
                    # ------------------------------------------------------------
                    metrics = compute_audio_metrics(output_path, transform.final_prompt)

                    # ------------------------------------------------------------
                    # Record Result Row
                    # ------------------------------------------------------------
                    rows.append(
                        {
                            "row_type": "per_audio",
                            "condition": condition,
                            "prompt_id": prompt_id,
                            "seed": seed,
                            "audio_path": str(output_path.relative_to(self.project_root)),
                            "raw_prompt": raw_prompt,
                            "prompt_text": transform.final_prompt,
                            "reasoning": transform.reasoning,
                            "reasoning_tags": transform.reasoning_tags,
                            "structured_json": json.dumps(transform.structured_json, ensure_ascii=False) if transform.structured_json else "",
                            "loudness_variance": metrics.loudness_variance,
                            "spectral_centroid_variance": metrics.spectral_centroid_variance,
                            "embedding_similarity": metrics.embedding_similarity,
                            "fad": metrics.fad,
                        }
                    )

        # ------------------------------------------------------------
        # Aggregate + Save Results
        # ------------------------------------------------------------
        summary_rows = summarize_results(rows)
        all_rows = rows + summary_rows
        write_results(all_rows, self.metrics_file)
        return all_rows


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config.yaml",
        help="Path to the experiment config file.",
    )
    parser.add_argument(
        "--prompt-cache",
        type=Path,
        default=None,
        help="Optional path to prepared prompt cache JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point for CLI execution.
    """
    args = parse_args()
    runner = ExperimentRunner(args.config)
    runner.run()


if __name__ == "__main__":
    main()
"""Run all Stable Audio Open experiment conditions end-to-end."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import yaml

from evaluate import compute_audio_metrics, summarize_results, write_results
from generate_audio import GenerationConfig, StableAudioGenerator
from qwen_prompt import build_transform


CONDITIONS = ("baseline", "structured", "cot")


class ExperimentRunner:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path.resolve()
        with self.config_path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.project_root = self.config_path.parent
        self.prompts = self._load_prompts(self.project_root / self.config["prompts_file"])
        self.output_root = self.project_root / self.config["outputs_root"]
        self.metrics_file = self.project_root / self.config["metrics_file"]
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
        with prompts_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def iter_enabled_conditions(self) -> Iterable[str]:
        condition_config = self.config.get("conditions", {})
        for condition in CONDITIONS:
            if condition_config.get(condition, {}).get("enabled", True):
                yield condition

    def build_seed(self, index: int) -> int:
        return int(self.config["seed_offset"]) + index

    def run(self) -> list[dict]:
        rows: list[dict] = []
        for prompt_record in self.prompts:
            prompt_id = prompt_record["id"]
            raw_prompt = prompt_record["prompt"]
            for condition in self.iter_enabled_conditions():
                transform = build_transform(raw_prompt, condition)
                condition_dir = self.output_root / condition
                for seed_index in range(int(self.config["num_seeds"])):
                    seed = self.build_seed(seed_index)
                    filename = f"{condition}_{prompt_id}_{seed}.wav"
                    output_path = condition_dir / filename
                    print(f"prompt_id={prompt_id} condition={condition} seed={seed}")
                    self.generator.generate_to_file(
                        prompt=transform.final_prompt,
                        seed=seed,
                        output_path=output_path,
                    )
                    metrics = compute_audio_metrics(output_path, transform.final_prompt)
                    rows.append(
                        {
                            "row_type": "per_audio",
                            "condition": condition,
                            "prompt_id": prompt_id,
                            "seed": seed,
                            "audio_path": str(output_path.relative_to(self.project_root)),
                            "prompt_text": transform.final_prompt,
                            "loudness_variance": metrics.loudness_variance,
                            "spectral_centroid_variance": metrics.spectral_centroid_variance,
                            "embedding_similarity": metrics.embedding_similarity,
                            "fad": metrics.fad,
                        }
                    )
        summary_rows = summarize_results(rows)
        all_rows = rows + summary_rows
        write_results(all_rows, self.metrics_file)
        return all_rows



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config.yaml",
        help="Path to the experiment config file.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    runner = ExperimentRunner(args.config)
    runner.run()


if __name__ == "__main__":
    main()

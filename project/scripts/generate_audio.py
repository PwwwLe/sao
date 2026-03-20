"""Audio generation helpers for Stable Audio Open experiments."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from einops import rearrange

try:
    from stable_audio_tools import get_pretrained_model
    from stable_audio_tools.inference.generation import generate_diffusion_cond
except ImportError:  # pragma: no cover - exercised only when dependency is missing
    get_pretrained_model = None
    generate_diffusion_cond = None


@dataclass
class GenerationConfig:
    model_name: str
    sample_rate: int
    duration_seconds: int
    device: str
    steps: int
    cfg_scale: float
    sigma_min: float
    sigma_max: float
    sampler_type: str


class StableAudioGenerator:
    """Thin wrapper around stable-audio-tools with deterministic seeding."""

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)
        self.model = None
        self.model_config = None
        self.sample_rate = config.sample_rate
        self.sample_size = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self) -> None:
        if get_pretrained_model is None or generate_diffusion_cond is None:
            raise ImportError(
                "stable-audio-tools is not installed. Install project/requirements.txt first."
            )
        if self.model is not None:
            return
        model, model_config = get_pretrained_model(self.config.model_name)
        self.model = model.to(self.device)
        self.model_config = model_config
        self.sample_rate = int(model_config.get("sample_rate", self.sample_rate))
        self.sample_size = int(model_config["sample_size"])

    def generate_to_file(self, prompt: str, seed: int, output_path: Path) -> Path:
        self.load()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _seed_everything(seed)

        conditioning = [{
            "prompt": prompt,
            "seconds_start": 0,
            "seconds_total": self.config.duration_seconds,
        }]

        audio = generate_diffusion_cond(
            self.model,
            steps=self.config.steps,
            cfg_scale=self.config.cfg_scale,
            conditioning=conditioning,
            sample_size=self.sample_size,
            sigma_min=self.config.sigma_min,
            sigma_max=self.config.sigma_max,
            sampler_type=self.config.sampler_type,
            device=self.device,
        )

        audio = rearrange(audio, "b d n -> d (b n)").to(torch.float32)
        peak = torch.max(torch.abs(audio))
        if peak > 0:
            audio = audio / peak
        sf.write(output_path, audio.transpose(0, 1).cpu().numpy(), self.sample_rate, subtype="PCM_16")
        return output_path



def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="Text prompt to render with Stable Audio Open.")
    parser.add_argument("--output", type=Path, required=True, help="Destination WAV file path.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for deterministic generation.")
    parser.add_argument("--duration", type=int, default=30, help="Audio duration in seconds.")
    parser.add_argument("--model-name", default="stabilityai/stable-audio-open-1.0")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=7.0)
    parser.add_argument("--sigma-min", type=float, default=0.3)
    parser.add_argument("--sigma-max", type=float, default=500.0)
    parser.add_argument("--sampler-type", default="dpmpp-3m-sde")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    generator = StableAudioGenerator(
        GenerationConfig(
            model_name=args.model_name,
            sample_rate=args.sample_rate,
            duration_seconds=args.duration,
            device=args.device,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            sampler_type=args.sampler_type,
        )
    )
    generator.generate_to_file(args.prompt, args.seed, args.output)


if __name__ == "__main__":
    main()

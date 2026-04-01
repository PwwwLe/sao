"""
Audio generation helpers for Stable Audio Open experiments.

This module provides a thin abstraction over stable-audio-tools
to support controlled, reproducible audio generation experiments.

Core responsibilities:
1. Load pretrained Stable Audio model (with HF cache awareness)
2. Provide deterministic generation via seeding
3. Wrap diffusion inference API
4. Save generated waveform to disk
"""

from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from einops import rearrange

# ------------------------------------------------------------
# Optional dependency import (graceful degradation)
# ------------------------------------------------------------
# Allows the script to fail cleanly if stable-audio-tools is missing.
try:
    from stable_audio_tools import get_pretrained_model
    from stable_audio_tools.inference.generation import generate_diffusion_cond
except ImportError:  # pragma: no cover
    get_pretrained_model = None
    generate_diffusion_cond = None

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
    run_id: str = "group2_structured_only",
) -> None:
    """
    Structured JSONL logging for experiment diagnostics.

    Designed for:
    - model loading debugging
    - network / HF cache inspection
    - reproducibility tracing

    Logging must NEVER interrupt generation pipeline.
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
            f.write(json_dumps(payload) + "\n")
    except Exception:
        pass


def json_dumps(payload: dict) -> str:
    """
    Local JSON serializer to avoid global import overhead.
    """
    import json as _json

    return _json.dumps(payload, ensure_ascii=False)


def _local_hf_model_config_exists(model_name: str) -> bool:
    """
    Check whether model_config.json exists in local HuggingFace cache.

    Used to diagnose:
    - offline mode failures
    - proxy misconfiguration
    - slow model downloads

    NOTE:
    This is a heuristic check (not guaranteed accurate).
    """
    if "/" not in model_name:
        return False
    org, name = model_name.split("/", 1)
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))
    hub_root = Path(hf_home) / "hub"
    snapshots_dir = hub_root / f"models--{org}--{name}" / "snapshots"
    if not snapshots_dir.is_dir():
        return False
    try:
        for p in snapshots_dir.rglob("model_config.json"):
            if p.is_file():
                return True
    except Exception:
        return False
    return False

# ------------------------------------------------------------
# Generation Configuration
# ------------------------------------------------------------
@dataclass
class GenerationConfig:
    """
    Configuration container for diffusion-based audio generation.

    These parameters directly map to Stable Audio diffusion controls.
    """
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
    """
    Wrapper around stable-audio-tools generation API.

    Key features:
    - Lazy model loading (load-on-first-use)
    - Device auto-selection (CUDA / MPS / CPU)
    - Deterministic seeding
    - Output normalization and saving
    """

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        
        # Resolve device automatically if needed
        self.device = self._resolve_device(config.device)
        
        # Model-related attributes (lazy initialization)
        self.model = None
        self.model_config = None
        
        self.sample_rate = config.sample_rate
        self.sample_size = None # determined after model load

    @staticmethod
    def _resolve_device(device: str) -> str:
        """
        Resolve computation device.

        Priority:
        CUDA > MPS (Apple Silicon) > CPU
        """
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self) -> None:
        """
        Load Stable Audio model (if not already loaded).

        Includes:
        - HF cache inspection
        - debug logging for download issues
        - model + config initialization
        """
        if get_pretrained_model is None or generate_diffusion_cond is None:
            raise ImportError(
                "stable-audio-tools is not installed. Install project/requirements.txt first."
            )
        
        # Prevent redundant loading
        if self.model is not None:
            return
        
        # ------------------------------------------------------------
        # Debug: check HF cache + proxy status
        # ------------------------------------------------------------
        _debug_log(
            hypothesis_id="H1_sao_model_download_blocking",
            location="scripts/generate_audio.py:StableAudioGenerator.load(before_get_pretrained_model)",
            message="Checking local HF cache presence for Stable Audio Open model",
            data={
                "model_name": self.config.model_name,
                "local_model_config_present": _local_hf_model_config_exists(self.config.model_name),
                "HF_HOME": os.environ.get("HF_HOME"),
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "HF_ENDPOINT": os.environ.get("HF_ENDPOINT"),
                "proxies": {
                    "http_proxy": os.environ.get("http_proxy"),
                    "HTTP_PROXY": os.environ.get("HTTP_PROXY"),
                    "https_proxy": os.environ.get("https_proxy"),
                    "HTTPS_PROXY": os.environ.get("HTTPS_PROXY"),
                    "all_proxy": os.environ.get("all_proxy"),
                    "ALL_PROXY": os.environ.get("ALL_PROXY"),
                },
            },
        )
        # #endregion
        try:
            model, model_config = get_pretrained_model(self.config.model_name)
        except Exception as exc:
            # #region agent log: sao download exception
            _debug_log(
                hypothesis_id="H1_sao_model_download_blocking",
                location="scripts/generate_audio.py:StableAudioGenerator.load(get_pretrained_model_failed)",
                message="Stable Audio Open get_pretrained_model failed",
                data={"error_type": type(exc).__name__, "error": str(exc)[:500]},
            )
            # #endregion
            raise
        
        # Move model to target device
        self.model = model.to(self.device)
        self.model_config = model_config
        
        # Override config from model metadata
        self.sample_rate = int(model_config.get("sample_rate", self.sample_rate))
        self.sample_size = int(model_config["sample_size"])

    def generate_to_file(self, prompt: str, seed: int, output_path: Path) -> Path:
        """
        Generate audio from text prompt and save to WAV file.

        Pipeline:
        1. Ensure model loaded
        2. Set random seed (full determinism)
        3. Build conditioning input
        4. Run diffusion sampling
        5. Normalize waveform
        6. Save as PCM16 WAV

        Args:
            prompt: final text prompt
            seed: random seed
            output_path: output file path

        Returns:
            Path to generated audio file
        """
        self.load()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure reproducibility
        _seed_everything(seed)

        # Build conditioning structure required by SAO
        conditioning = [{
            "prompt": prompt,
            "seconds_start": 0,
            "seconds_total": self.config.duration_seconds,
        }]

        # ------------------------------------------------------------
        # Diffusion Sampling
        # ------------------------------------------------------------
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

        # ------------------------------------------------------------
        # Post-processing
        # ------------------------------------------------------------
        # Reshape: (batch, channel, samples) → (channel, samples)
        audio = rearrange(audio, "b d n -> d (b n)").to(torch.float32)
        
        # Peak normalization
        peak = torch.max(torch.abs(audio))
        if peak > 0:
            audio = audio / peak
            
        # Save audio
        sf.write(output_path, audio.transpose(0, 1).cpu().numpy(), self.sample_rate, subtype="PCM_16")
        return output_path



def _seed_everything(seed: int) -> None:
    """
    Set all random seeds for reproducibility.

    Covers:
    - Python random
    - NumPy
    - PyTorch (CPU + CUDA)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ------------------------------------------------------------
# CLI Interface
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for standalone usage.
    """
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
    """
    CLI entry point.

    Enables quick testing:
    python generate_audio.py --prompt "rain forest ambience"
    """
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

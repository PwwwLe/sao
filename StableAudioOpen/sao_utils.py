"""Stable Audio Open generation utilities shared by experiment frontends."""

import torch
import torchaudio
import soundfile as sf
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond


class SAOGenerator:
    """Lazy wrapper around the Stable Audio Open model and generation API."""

    def __init__(self, device="auto"):
        """Initialize the SAO model on the selected device.

        Args:
            device: Device string or "auto" for CUDA/MPS/CPU auto-selection.
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.model, self.config = get_pretrained_model(
            "stabilityai/stable-audio-open-1.0"
        )
        self.model = self.model.to(device)
        self.sample_rate = self.config["sample_rate"]
        self.sample_size = self.config["sample_size"]

    def generate(
        self,
        prompt: str,
        seconds: int,
        seed: int,
        out_path: str,
        steps: int = 100,
        cfg_scale: float = 7.0,
        sigma_min: float = 0.3,
        sigma_max: float = 500.0,
        sampler_type: str = "dpmpp-3m-sde",
    ):
        """Generate audio with SAO and save it to disk.

        Args:
            prompt: Final SAO prompt string.
            seconds: Audio duration in seconds.
            seed: Random seed for deterministic generation.
            out_path: Output WAV file path.
            steps: Diffusion sampling steps.
            cfg_scale: Classifier-free guidance scale.
            sigma_min: Minimum sigma for sampler.
            sigma_max: Maximum sigma for sampler.
            sampler_type: Sampler algorithm name.
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        conditioning = [{
            "prompt": prompt,
            "seconds_start": 0,
            "seconds_total": seconds
        }]

        audio = generate_diffusion_cond(
            self.model,
            steps=steps,
            cfg_scale=cfg_scale,
            conditioning=conditioning,
            sample_size=self.sample_size,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sampler_type=sampler_type,
            device=self.device
        )

        audio = rearrange(audio, "b d n -> d (b n)")
        # Normalize to int16 PCM to keep output compatible with common audio players.
        audio = (
            audio.to(torch.float32)
            .div(torch.max(torch.abs(audio)))
            .clamp(-1, 1)
            .mul(32767)
            .to(torch.int16)
            .cpu()
        )

        self._save_audio(out_path, audio)

    def _save_audio(self, out_path: str, audio: torch.Tensor) -> None:
        """Save generated audio with torchaudio and fallback to soundfile.

        Args:
            out_path: Destination WAV file path.
            audio: Audio tensor in shape [channels, samples].
        """
        # torchaudio>=2.8 may require torchcodec for save(); fallback to soundfile.
        try:
            torchaudio.save(out_path, audio, self.sample_rate)
            return
        except (ImportError, RuntimeError):
            pass

        # soundfile expects shape [num_samples, num_channels].
        audio_np = audio.transpose(0, 1).numpy()
        sf.write(out_path, audio_np, self.sample_rate, subtype="PCM_16")

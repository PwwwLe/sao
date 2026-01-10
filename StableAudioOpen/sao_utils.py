import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond


class SAOGenerator:
    def __init__(self, device="cuda"):
        self.device = device
        self.model, self.config = get_pretrained_model(
            "stabilityai/stable-audio-open-1.0"
        )
        self.model = self.model.to(device)
        self.sample_rate = self.config["sample_rate"]
        self.sample_size = self.config["sample_size"]

    def generate(self, prompt: str, seconds: int, seed: int, out_path: str):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        conditioning = [{
            "prompt": prompt,
            "seconds_start": 0,
            "seconds_total": seconds
        }]

        audio = generate_diffusion_cond(
            self.model,
            steps=100,
            cfg_scale=7,
            conditioning=conditioning,
            sample_size=self.sample_size,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
            device=self.device
        )

        audio = rearrange(audio, "b d n -> d (b n)")
        audio = (
            audio.to(torch.float32)
            .div(torch.max(torch.abs(audio)))
            .clamp(-1, 1)
            .mul(32767)
            .to(torch.int16)
            .cpu()
        )

        torchaudio.save(out_path, audio, self.sample_rate)

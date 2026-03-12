"""Standalone SAO reproduction script for quick local validation."""

import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from prompt_linearizer import linearize_structured_prompt

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pretrained Stable Audio Open model and configuration.
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

model = model.to(device)

# Build generation conditioning payload.
prompt = "Format: Solo | Genre: Foley | Sub-genre: UI SFX | Instruments: pure sine wave ping, metallic digital click, slight resonant tail | Moods: clean, precise, high-tech | Styles: Video Games, High Tech, Sci-Fi | Tempo: Medium | BPM: 120 | Details: 0.5 second one-shot, snappy envelope, decay quickly | Quality: high-quality, stereo"

conditioning = [{
    "prompt": prompt,
    "seconds_start": 0,
    "seconds_total": 30
}]


# Generate stereo waveform.
output = generate_diffusion_cond(
    model,
    steps=100,
    cfg_scale=7,
    conditioning=conditioning,
    sample_size=sample_size,
    sigma_min=0.3,
    sigma_max=500,
    sampler_type="dpmpp-3m-sde",
    device=device
)

# Merge batch dimension into a single continuous sample sequence.
output = rearrange(output, "b d n -> d (b n)")

# Peak normalize, clip, convert to int16, and save to file
output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
torchaudio.save("output-1.wav", output, sample_rate)

import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from prompt_linearizer import linearize_structured_prompt

device = "cuda" if torch.cuda.is_available() else "cpu"

import requests
import json

QWEN_SERVICE_URL = "http://127.0.0.1:8008/refine_prompt"


def get_structured_prompt(raw_prompt: str) -> str:
    payload = {
        "raw_prompt": raw_prompt,
        "max_new_tokens": 512
    }
    resp = requests.post(QWEN_SERVICE_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["structured_prompt"]


# Download model
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

model = model.to(device)

# Set up text and timing conditioning
raw_prompt = "A luxurious Indietronica instrumental perfect for a perfume advertisement"

structured_prompt = get_structured_prompt(raw_prompt)

final_prompt = linearize_structured_prompt(structured_prompt)

conditioning = [{
    "prompt": final_prompt,
    "seconds_start": 0,
    "seconds_total": 30
}]


# Generate stereo audio
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

# Rearrange audio batch to a single sequence
output = rearrange(output, "b d n -> d (b n)")

# Peak normalize, clip, convert to int16, and save to file
output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
torchaudio.save("output-1.wav", output, sample_rate)

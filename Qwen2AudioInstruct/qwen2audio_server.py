import torch
import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# =========================
# 1. Load Model
# =========================

MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen2-Audio-7B-Instruct")
LOCAL_MODEL_DIR = os.environ.get("QWEN_LOCAL_MODEL_DIR", "").strip()
USE_LOCAL_MODEL = bool(LOCAL_MODEL_DIR)

if USE_LOCAL_MODEL and not os.path.isdir(LOCAL_MODEL_DIR):
    raise RuntimeError(
        f"QWEN_LOCAL_MODEL_DIR does not exist: {LOCAL_MODEL_DIR}"
    )

MODEL_SOURCE = LOCAL_MODEL_DIR if USE_LOCAL_MODEL else MODEL_ID
print(f"[QWEN] Loading model from: {MODEL_SOURCE}")
print(f"[QWEN] local_files_only={USE_LOCAL_MODEL}")

processor = AutoProcessor.from_pretrained(
    MODEL_SOURCE,
    local_files_only=USE_LOCAL_MODEL,
)
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    MODEL_SOURCE,
    torch_dtype="auto",
    device_map="auto",
    local_files_only=USE_LOCAL_MODEL,
)
model.eval()

# =========================
# 2. Prompt Schema & System Prompt (Revised for SAO Prompt-Compiler)
# =========================

DEFAULT_PROMPT_SCHEMA = """
{
  "audio_type": "",

    "format": "",
    "genre": "",
    "sub_genre": "",

    "instruments_primary": "",
    "instruments_supporting": "",
    "rhythm_components": "",
    "texture_elements": "",

    "moods": "",
    "styles": "",

    "tempo": "",
    "bpm": "",

    "details": "",
  "production": "",
  "use_case": "",

    "looping": "",
    "duration_hint": "",

    "negative_prompt": "",

    "sao_prompt": ""
}
"""

DEFAULT_SYSTEM_PROMPT_TEMPLATE = """
You are a professional Stable Audio Open (SAO) prompt engineer and prompt-compiler.

CRITICAL REQUIREMENTS (MUST FOLLOW):
- Output ONLY valid JSON. No explanations, no markdown, no extra keys.
- ALL text values in the output JSON MUST be written in ENGLISH.
- If the user input is not in English, translate it into English internally before filling the JSON.
- Do NOT invent facts. If a field is not inferable from the user input, leave it as an empty string.
- Follow Stable Audio prompt best practices: clear genre/sub-genre, specific moods, explicit instruments, and production characteristics.
- The SAO prompt must be a SINGLE LINE string using the exact field order and separators shown below.

GOAL:
Convert the user's raw request into a structured, controllable SAO input prompt suitable for game audio generation.

AUDIO TYPES:
- "BGM" (music track)
- "AMBIENCE" (environmental soundscape)
- "UI_SFX" (UI/interaction one-shots)
- "FOLEY_SFX" (short sound effects / events)

FORMAT RULES:
- If audio_type is "BGM": format = "Band"
- If audio_type is "AMBIENCE" or "UI_SFX" or "FOLEY_SFX": format = "Solo"

FIELD GUIDELINES (FILL WHEN INFERABLE):
- genre/sub_genre: be specific; sub_genre should refine genre.
- moods: use precise, music-audio-relevant adjectives (e.g., "euphoric", "melancholic", "mystical", "tense", "adventurous").
- styles: include "Video Games" when applicable; add setting tags if explicitly implied (e.g., "Sci-Fi", "Fantasy", "Retro", "High Tech", "Cinematic").
- instruments_*:
    - BGM: separate primary, supporting, rhythm_components, and texture_elements.
    - UI_SFX/FOLEY_SFX: use sound design terms (e.g., "sine ping", "FM blip", "resonant click", "noise burst", "metallic tick").
    - AMBIENCE: describe sound sources (e.g., "field recording rain", "wind", "distant thunder", "room tone") and texture_elements.
- tempo/bpm:
    - If the user provides BPM, copy it.
    - If BPM is not provided, leave bpm empty (do NOT guess).
- looping:
    - For AMBIENCE or loop requests, set looping to "loopable, seamless loop" when explicitly requested.
    - Otherwise leave empty.
- duration_hint:
    - If the user gives a duration, copy it (e.g., "0.5 seconds", "30 seconds loop", "2 minutes").
    - Otherwise leave empty.
- negative_prompt:
    - Only include if the user asks for exclusions or quality constraints; otherwise leave empty.

SAO PROMPT CONSTRUCTION:
Construct "sao_prompt" as a SINGLE LINE using this exact order and separator:
Format: <format> | Genre: <genre> | Sub-genre: <sub_genre> | Instruments: <merged_instruments> | Moods: <moods> | Styles: <styles> | Tempo: <tempo> | BPM: <bpm> | Details: <details_merged> | Quality: high-quality, stereo

Where:
- <merged_instruments> = join non-empty parts in this order:
    instruments_primary; instruments_supporting; rhythm_components; texture_elements
- <details_merged> should compactly include (when non-empty):
    use_case; production; looping; duration_hint; any additional details
- If a field is empty, keep the label but leave the value blank (do NOT remove labels).
    Example: "BPM:  |"

OUTPUT JSON SCHEMA:
{prompt_schema}
"""

# =========================
# 3. FastAPI Initialization
# =========================

app = FastAPI(
    title="Qwen2-Audio Prompt Compiler",
    version="1.0"
)

# =========================
# 4. 请求体定义
# =========================

class PromptRequest(BaseModel):
    raw_prompt: str
    max_new_tokens: int = 512
    system_prompt: str | None = None
    prompt_schema: str | None = None

class PromptResponse(BaseModel):
    structured_prompt: str


# =========================
# 5. 核心推理函数
# =========================

def build_system_prompt(
    system_prompt: str | None = None,
    prompt_schema: str | None = None
) -> str:
    schema = prompt_schema or DEFAULT_PROMPT_SCHEMA
    if system_prompt:
        return system_prompt
    return DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(prompt_schema=schema)


def refine_prompt(
    raw_prompt: str,
    max_new_tokens=512,
    system_prompt: str | None = None,
    prompt_schema: str | None = None
) -> str:
    resolved_system_prompt = build_system_prompt(
        system_prompt=system_prompt,
        prompt_schema=prompt_schema
    )

    conversation = [
    {"role": "system", "content": resolved_system_prompt},
    {
        "role": "user",
        "content": f"""
    IMPORTANT: Respond in ENGLISH ONLY.

    User prompt:
    {raw_prompt}
    """
        },
    ]

    text = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(text=text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,  
        )

    gen = output_ids[:, inputs["input_ids"].shape[1]:]
    out = processor.batch_decode(
        gen,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return out


# =========================
# 6. HTTP 接口
# =========================

@app.post("/refine_prompt", response_model=PromptResponse)
def refine_prompt_api(req: PromptRequest):
    structured = refine_prompt(
        raw_prompt=req.raw_prompt,
        max_new_tokens=req.max_new_tokens,
        system_prompt=req.system_prompt,
        prompt_schema=req.prompt_schema
    )
    return {"structured_prompt": structured}

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# =========================
# 1. Load Model
# =========================

model_id = "Qwen/Qwen2-Audio-7B-Instruct"

processor = AutoProcessor.from_pretrained(model_id)
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto"
)
model.eval()

# =========================
# 2. Prompt Schema & System Prompt
# =========================

PROMPT_SCHEMA = """
{
  "audio_type": "",

  "sound_source": "",
  "sound_event": "",
  "environment": "",

  "style": "",
  "mood": "",
  "tempo": "",
  "rhythm": "",

  "texture": "",
  "dynamics": "",
  "spatial": "",

  "structure": "",
  "production": "",
  "use_case": "",
  "negative": ""
}
"""

SYSTEM_PROMPT = f"""
You are a professional audio prompt engineer for Stable Audio Open.

CRITICAL REQUIREMENTS (MUST FOLLOW):
- ALL text values in the output JSON MUST be written in ENGLISH.
- Do NOT output any non-English words.
- If the input is not in English, TRANSLATE it into English first.
- Output ONLY valid JSON. No explanations, no markdown.

TASK INSTRUCTIONS:
1. Infer the audio_type of the input audio.
2. Describe the audio as a sound scene, not as text or labels.
3. Use concrete auditory attributes (texture, dynamics, spatial).
4. Only include musical concepts when the audio is music.

Schema:
{PROMPT_SCHEMA}
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

class PromptResponse(BaseModel):
    structured_prompt: str


# =========================
# 5. 核心推理函数
# =========================

def refine_prompt(raw_prompt: str, max_new_tokens=512) -> str:
    conversation = [
    {"role": "system", "content": SYSTEM_PROMPT},
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
        max_new_tokens=req.max_new_tokens
    )
    return {"structured_prompt": structured}

import json
import csv
import os
import requests
from json import dumps

from sao_utils import SAOGenerator
from prompt_linearizer import linearize_structured_prompt
from json_sanitizer import extract_json_block

# =========================
# Global Config
# =========================

QWEN_SERVICE_URL = "http://127.0.0.1:8008/refine_prompt"

PROMPT_FILE = "../experiments/prompts.jsonl"
OUT_ROOT = "../experiments/results"
LEDGER_PATH = "../experiments/prompt_ledger.jsonl"

SEED = 42
DURATION = 30

# Only two experimental modes
MODES = ["raw", "structured_nl"]

# =========================
# Helper Functions
# =========================

def get_structured_prompt(raw_prompt: str) -> str:
    payload = {
        "raw_prompt": raw_prompt,
        "max_new_tokens": 512
    }
    resp = requests.post(QWEN_SERVICE_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["structured_prompt"]


def load_prompts(path):
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


def ensure_dirs():
    for mode in MODES:
        os.makedirs(os.path.join(OUT_ROOT, mode), exist_ok=True)

# =========================
# Main Experiment Logic
# =========================

def main():
    ensure_dirs()
    prompts = load_prompts(PROMPT_FILE)
    generator = SAOGenerator()

    # ---------- Open files ONCE ----------
    with open(LEDGER_PATH, "w", encoding="utf-8") as ledger_f, \
         open(os.path.join(OUT_ROOT, "metadata.csv"), "w", newline="", encoding="utf-8") as meta_f:

        writer = csv.writer(meta_f)
        writer.writerow([
            "prompt_id",
            "mode",
            "seed",
            "raw_prompt",
            "final_prompt",
            "wav_path"
        ])

        # ---------- Loop over prompts ----------
        for item in prompts:
            pid = item["id"]
            raw_prompt = item["prompt"]

            # ===== Step 1: Qwen Prompt Compilation =====
            structured_prompt_raw = get_structured_prompt(raw_prompt)

            # ===== Step 2: Parse & Linearize (Structured → NL) =====
            try:
                structured_json = extract_json_block(structured_prompt_raw)
                linearized_prompt = linearize_structured_prompt(structured_prompt_raw)
            except Exception as e:
                print(f"[WARN] Prompt {pid} parse failed: {e}")
                structured_json = None
                linearized_prompt = raw_prompt  # fallback to raw

            # ===== Step 3: Write Prompt Ledger (ONCE per prompt) =====
            ledger_entry = {
                "prompt_id": pid,
                "raw_prompt": raw_prompt,
                "structured_prompt": structured_json,
                "structured_prompt_raw_text": structured_prompt_raw,
                "linearized_prompt": linearized_prompt
            }

            ledger_f.write(dumps(ledger_entry, ensure_ascii=False) + "\n")
            ledger_f.flush()

            # ===== Step 4: Two Experimental Modes =====
            for mode in MODES:
                if mode == "raw":
                    final_prompt = raw_prompt
                elif mode == "structured_nl":
                    final_prompt = linearized_prompt
                else:
                    continue

                out_name = f"prompt{pid:02d}_seed{SEED}.wav"
                out_path = os.path.join(OUT_ROOT, mode, out_name)

                print(f"[{mode}] Prompt {pid} → {out_path}")

                generator.generate(
                    prompt=final_prompt,
                    seconds=DURATION,
                    seed=SEED,
                    out_path=out_path
                )

                writer.writerow([
                    pid,
                    mode,
                    SEED,
                    raw_prompt,
                    final_prompt.replace("\n", " "),
                    out_path
                ])

# =========================
# Entry Point
# =========================

if __name__ == "__main__":
    main()

# Stable Audio Open experiment pipeline

This project runs three controlled text-to-audio conditions for ambience generation:

1. `baseline`: raw prompt -> Stable Audio Open.
2. `structured`: raw prompt -> simulated Qwen2-Audio structured prompt -> Stable Audio Open.
3. `cot`: raw prompt -> simulated Qwen2-Audio chain-of-thought rewrite -> Stable Audio Open.

## Setup

```bash
cd project
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Files

- `config.yaml`: central experiment settings.
- `prompts/raw_prompts.json`: 8 ambience prompts.
- `scripts/qwen_prompt.py`: simulated structured and CoT prompt generation.
- `scripts/generate_audio.py`: Stable Audio Open generator wrapper and CLI.
- `scripts/evaluate.py`: custom metrics plus optional stable-audio-metrics hooks.
- `scripts/run_experiments.py`: end-to-end runner for all conditions and seeds.

## Run the full experiment

```bash
cd project
python scripts/run_experiments.py --config config.yaml
```

Outputs are written to:

- `outputs/baseline/`
- `outputs/structured/`
- `outputs/cot/`
- `metrics/results.csv`

## Generate a single file manually

```bash
cd project
python scripts/generate_audio.py \
  --prompt "A peaceful forest ambience at dawn with soft birdsong and a distant creek." \
  --output outputs/baseline/sample.wav \
  --seed 1000
```

## Evaluate an existing manifest

Create a JSON manifest containing the per-audio rows and then run:

```bash
cd project
python scripts/evaluate.py --manifest metrics/manifest.json --output metrics/results.csv
```

## Notes

- Reproducibility is controlled with `seed_offset` and `num_seeds` in `config.yaml`.
- `stable-audio-metrics` support differs by version, so CLAP similarity is attempted when available and FAD is treated as optional.
- Output file names follow `{condition}_{prompt_id}_{seed}.wav`.

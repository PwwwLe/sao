"""Metrics for Stable Audio Open experiments."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

try:
    from stable_audio_metrics import clap_score
except ImportError:  # pragma: no cover
    clap_score = None


@dataclass
class AudioMetrics:
    loudness_variance: float
    spectral_centroid_variance: float
    embedding_similarity: float | None
    fad: float | None



def rms_loudness_variance(audio: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> float:
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    return float(np.var(rms))



def spectral_centroid_variance(audio: np.ndarray, sample_rate: int, hop_length: int = 512) -> float:
    centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate, hop_length=hop_length)[0]
    return float(np.var(centroids))



def embedding_similarity_score(audio_path: Path, prompt: str) -> float | None:
    if clap_score is None:
        return None
    score = clap_score([str(audio_path)], [prompt])
    if isinstance(score, dict):
        for key in ("mean", "clap_score", "score"):
            if key in score:
                return float(score[key])
    if isinstance(score, (list, tuple, np.ndarray)):
        return float(np.asarray(score).mean())
    return float(score)



def fad_score(audio_paths: Iterable[Path], background_dir: Path | None = None) -> float | None:
    if background_dir is None:
        return None

    candidate_locations = [
        ("stable_audio_metrics.fad", "FrechetAudioDistance"),
        ("stable_audio_metrics.metrics.fad", "FrechetAudioDistance"),
    ]
    for module_name, attr_name in candidate_locations:
        try:
            module = importlib.import_module(module_name)
            fad_cls = getattr(module, attr_name)
            metric = fad_cls()
            return float(metric.score([str(path) for path in audio_paths], str(background_dir)))
        except Exception:
            continue
    return None



def compute_audio_metrics(audio_path: Path, prompt: str) -> AudioMetrics:
    audio, sample_rate = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return AudioMetrics(
        loudness_variance=rms_loudness_variance(audio),
        spectral_centroid_variance=spectral_centroid_variance(audio, sample_rate),
        embedding_similarity=embedding_similarity_score(audio_path, prompt),
        fad=None,
    )



def summarize_results(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    numeric_columns = [
        "loudness_variance",
        "spectral_centroid_variance",
        "embedding_similarity",
        "fad",
    ]
    summary_rows: list[dict] = []
    grouped = frame.groupby(["condition", "prompt_id"], dropna=False)
    for (condition, prompt_id), group in grouped:
        summary: dict[str, object] = {
            "row_type": "prompt_summary",
            "condition": condition,
            "prompt_id": prompt_id,
            "seed": "all",
            "audio_path": "",
            "prompt_text": group["prompt_text"].iloc[0],
        }
        for column in numeric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            summary[f"{column}_mean"] = float(values.mean()) if values.notna().any() else None
            summary[f"{column}_variance"] = float(values.var(ddof=0)) if values.notna().any() else None
        summary_rows.append(summary)
    return summary_rows



def write_results(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_type",
        "condition",
        "prompt_id",
        "seed",
        "audio_path",
        "prompt_text",
        "loudness_variance",
        "spectral_centroid_variance",
        "embedding_similarity",
        "fad",
        "loudness_variance_mean",
        "loudness_variance_variance",
        "spectral_centroid_variance_mean",
        "spectral_centroid_variance_variance",
        "embedding_similarity_mean",
        "embedding_similarity_variance",
        "fad_mean",
        "fad_variance",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSON manifest with per-audio experiment rows.")
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV for detailed and summary metrics.")
    parser.add_argument("--background-dir", type=Path, default=None, help="Optional reference directory for FAD.")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.background_dir:
        grouped_audio: dict[tuple[str, str], list[Path]] = {}
        for row in rows:
            key = (row["condition"], row["prompt_id"])
            grouped_audio.setdefault(key, []).append(Path(row["audio_path"]))
        for row in rows:
            row["fad"] = fad_score(grouped_audio[(row["condition"], row["prompt_id"])], args.background_dir)
    write_results(rows + summarize_results(rows), args.output)


if __name__ == "__main__":
    main()

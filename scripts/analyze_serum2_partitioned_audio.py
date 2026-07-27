#!/usr/bin/env python3
"""Render and compare the five reconstructed Serum 2 state fixtures against init."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.platform_env import ENV
from core.plugin_host import make_dawdreamer_processor, render_dawdreamer_note


ARTIFACT_DIR = PROJECT_ROOT / "data" / "models" / "serum2_partitioned_audio"
REPORT_PATH = PROJECT_ROOT / "data" / "models" / "serum2_partitioned_audio_report.json"
FIXTURES = (
    ("KillerRattle", 13, "1.vstpreset"),
    ("BAM Just Hold E", 15, "2.vstpreset"),
    ("QUIX Lead RPS 2", 3, "3.vstpreset"),
    ("Fixture Bass Growl", 26, "4.vstpreset"),
    ("Fixture Lead Arpeggio", 21, "5.vstpreset"),
)


def serum2_candidate() -> Any:
    return next(item for item in ENV.plugins_for("serum2") if item.format == "VST3")


def render_worker(container: Path | None, output: Path) -> dict[str, Any]:
    engine, processor = make_dawdreamer_processor(serum2_candidate())
    loaded = None
    if container is not None:
        loaded = processor.load_vst3_preset(str(container.resolve()))
        if loaded is False:
            raise RuntimeError("DawDreamer load_vst3_preset returned False")
    audio = render_dawdreamer_note(engine, processor, midi_note=60, duration=4.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, audio.T, 44_100, subtype="FLOAT")
    return {"load_return": loaded, "samples": int(audio.shape[-1]), "wav": str(output)}


def isolated_render(container: Path | None, output: Path) -> dict[str, Any]:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--output", str(output)]
    if container is not None:
        command.extend(("--container", str(container)))
    completed = subprocess.run(command, capture_output=True, text=True, timeout=90)
    result: dict[str, Any] = {
        "exit_code": completed.returncode,
        "crashed_by_signal": -completed.returncode if completed.returncode < 0 else None,
        "stderr": completed.stderr,
    }
    for line in completed.stdout.splitlines():
        if line.startswith("WORKER_RESULT="):
            result["result"] = json.loads(line.split("=", 1)[1])
    if completed.returncode or "result" not in result:
        raise RuntimeError(f"isolated render failed: {result}")
    return result


def audio_features(path: Path) -> dict[str, Any]:
    audio, sample_rate = librosa.load(path, sr=44_100, mono=True)
    stft = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
    centroid = librosa.feature.spectral_centroid(S=stft, sr=sample_rate)[0]
    flatness = librosa.feature.spectral_flatness(S=stft)[0]
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20, n_fft=2048, hop_length=512)
    onset = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=512)
    embedding = np.concatenate((mfcc.mean(axis=1), mfcc.std(axis=1))).astype(np.float64)
    norm = float(np.linalg.norm(embedding))
    if norm:
        embedding /= norm
    return {
        "centroid_mean_hz": float(np.mean(centroid)),
        "centroid_std_hz": float(np.std(centroid)),
        "flatness_mean": float(np.mean(flatness)),
        "flatness_std": float(np.std(flatness)),
        "mfcc_temporal_delta": float(np.mean(np.abs(np.diff(mfcc, axis=1)))),
        "onset_strength_mean": float(np.mean(onset)),
        "rms_dbfs": float(20.0 * np.log10(max(float(np.sqrt(np.mean(audio * audio))), 1e-12))),
        "embedding": embedding.tolist(),
    }


def pairwise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for left, right in combinations(rows, 2):
        distance = float(cosine(left["embedding"], right["embedding"]))
        result.append({"left": left["name"], "right": right["name"], "mfcc_cosine_distance": distance})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--container", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.output is None:
            parser.error("--output is required in worker mode")
        try:
            result = render_worker(args.container, args.output)
            print("WORKER_RESULT=" + json.dumps(result), flush=True)
            return 0
        except Exception as exc:
            print(
                "WORKER_RESULT=" + json.dumps({"type": type(exc).__name__, "error": repr(exc)}),
                flush=True,
            )
            return 2

    source_dir = PROJECT_ROOT / "data" / "models" / "serum2_partitioned_spike"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    render_reports = []
    preset_rows = []
    for index, (name, active_slots, filename) in enumerate(FIXTURES, start=1):
        wav = ARTIFACT_DIR / f"preset_{index}.wav"
        render_reports.append(isolated_render(source_dir / filename, wav))
        features = audio_features(wav)
        preset_rows.append({"name": name, "active_mod_slots": active_slots, "wav": str(wav), **features})
        print(f"AUDIO_PROGRESS=preset {index}/5", flush=True)

    init_rows = []
    for index in range(1, 4):
        wav = ARTIFACT_DIR / f"init_{index}.wav"
        render_reports.append(isolated_render(None, wav))
        init_rows.append({"name": f"Init {index}", "wav": str(wav), **audio_features(wav)})
        print(f"AUDIO_PROGRESS=init {index}/3", flush=True)

    preset_pairs = pairwise(preset_rows)
    init_pairs = pairwise(init_rows)
    preset_init = [
        {
            "preset": preset["name"],
            "init": init["name"],
            "mfcc_cosine_distance": float(cosine(preset["embedding"], init["embedding"])),
        }
        for preset in preset_rows
        for init in init_rows
    ]
    slots = np.asarray([row["active_mod_slots"] for row in preset_rows], dtype=np.float64)
    correlations = {}
    for metric in (
        "centroid_std_hz",
        "flatness_mean",
        "mfcc_temporal_delta",
        "onset_strength_mean",
    ):
        value = spearmanr(slots, [row[metric] for row in preset_rows])
        correlations[metric] = {"spearman_r": float(value.statistic), "p_value": float(value.pvalue)}

    init_ceiling = max((row["mfcc_cosine_distance"] for row in init_pairs), default=0.0)
    min_preset_pair = min(row["mfcc_cosine_distance"] for row in preset_pairs)
    min_preset_init = min(row["mfcc_cosine_distance"] for row in preset_init)
    tolerance = max(init_ceiling * 3.0, 1e-5)
    payload = {
        "host": "DawDreamer VST3",
        "render_spec": {"sample_rate": 44_100, "midi_note": 60, "duration_s": 4.0},
        "presets": preset_rows,
        "init_repeats": init_rows,
        "preset_pair_distances": preset_pairs,
        "init_pair_distances": init_pairs,
        "preset_init_distances": preset_init,
        "complexity_correlations": correlations,
        "decision_metrics": {
            "max_init_repeat_distance": init_ceiling,
            "min_preset_pair_distance": min_preset_pair,
            "min_preset_init_distance": min_preset_init,
            "distinctness_tolerance": tolerance,
            "all_presets_distinct_from_each_other": min_preset_pair > tolerance,
            "all_presets_distinct_from_init": min_preset_init > tolerance,
        },
        "render_processes": render_reports,
    }
    REPORT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["decision_metrics"], indent=2))
    print(f"OUTPUT={REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

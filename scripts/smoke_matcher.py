#!/usr/bin/env python3
"""Run one bounded in-library matcher smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.dataset import FEATURE_DIR
from core.matcher import AnalysisBySynthesisMatcher, SearchConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("preset_id", type=int)
    parser.add_argument("--synth", choices=("serum1", "serum2"), required=True)
    parser.add_argument("--budget", type=int, default=64)
    args = parser.parse_args()
    manifest = np.load(FEATURE_DIR / "note_manifest.npz")
    row = next(
        index
        for index, (preset_id, note) in enumerate(
            zip(manifest["preset_ids"], manifest["midi_notes"], strict=True)
        )
        if int(preset_id) == args.preset_id and int(note) == 60
    )
    audio, sample_rate = sf.read(str(manifest["wav_paths"][row]), dtype="float32", always_2d=True)
    matcher = AnalysisBySynthesisMatcher(processes=4)
    try:
        result = matcher.match(
            np.mean(audio, axis=1),
            sample_rate,
            synth_hint=args.synth,
            config=SearchConfig(max_evaluations=args.budget, max_seconds=120.0),
        )
    finally:
        matcher.close()
    report = {
        "query_preset_id": args.preset_id,
        "detected_midi_note": result.midi_note,
        "acoustic_midi_note": result.acoustic_midi_note,
        "retrieved": result.retrieved_preset_ids,
        "best_base_preset_id": result.best.base_preset_id,
        "best_origin": result.best.origin,
        "objective": result.best.objective,
        "stft_loss": result.best.stft_loss,
        "clap_cosine": result.best.clap_cosine,
        "evaluations": result.evaluations,
        "elapsed_s": result.elapsed_s,
        "evaluations_per_second": result.evaluations_per_second,
    }
    print("MATCHER_SMOKE=" + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Recompute a bounded set of handcrafted rows after a feature-definition change.

CLAP embeddings are intentionally left untouched.  This script exists so a
resumable feature run can repair only affected handcrafted rows without paying
the cost of embedding those audio files again.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.features import HANDCRAFTED_NAMES, handcrafted_features, load_audio_48k_mono


FEATURE_DIR = PROJECT_ROOT / "data" / "features"
NOTE_FEATURES = FEATURE_DIR / "note_handcrafted.npy"
NOTE_COMPLETE = FEATURE_DIR / "note_complete.npy"
NOTE_MANIFEST = FEATURE_DIR / "note_manifest.npz"
PRESET_FEATURES = FEATURE_DIR / "preset_handcrafted.npy"
PRESET_MANIFEST = FEATURE_DIR / "preset_manifest.npz"
REPORT = PROJECT_ROOT / "data" / "models" / "handcrafted_repair_report.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, required=True)
    args = parser.parse_args()
    manifest = np.load(NOTE_MANIFEST)
    paths = manifest["wav_paths"]
    preset_ids = manifest["preset_ids"]
    complete = np.load(NOTE_COMPLETE, mmap_mode="r")
    features = np.lib.format.open_memmap(NOTE_FEATURES, mode="r+")
    start = max(0, args.start)
    stop = min(args.stop, len(paths))
    if start >= stop:
        raise ValueError(f"Empty repair range [{start}, {stop})")
    incomplete = np.flatnonzero(~complete[start:stop])
    if incomplete.size:
        raise RuntimeError("Cannot repair rows that have not completed feature extraction")

    for index in range(start, stop):
        prepared = load_audio_48k_mono(Path(str(paths[index])))
        features[index] = handcrafted_features(prepared.waveform, prepared.sample_rate)
        if (index - start + 1) % 25 == 0 or index + 1 == stop:
            features.flush()
            print(
                "REPAIR_PROGRESS="
                + json.dumps({"completed": index - start + 1, "total": stop - start}),
                flush=True,
            )

    # The full analysis pass may already have produced preset-level means.
    # Refresh them in place so both levels use the repaired values.
    preset_rows_updated = 0
    if PRESET_FEATURES.exists() and PRESET_MANIFEST.exists():
        preset_manifest = np.load(PRESET_MANIFEST)
        preset_level_ids = preset_manifest["preset_ids"]
        preset_features = np.lib.format.open_memmap(PRESET_FEATURES, mode="r+")
        affected = set(int(value) for value in preset_ids[start:stop])
        for preset_row, preset_id in enumerate(preset_level_ids):
            numeric_id = int(preset_id)
            if numeric_id not in affected:
                continue
            note_rows = np.flatnonzero(preset_ids == numeric_id)
            preset_features[preset_row] = np.mean(features[note_rows], axis=0)
            preset_rows_updated += 1
        preset_features.flush()

    report = {
        "range_start": start,
        "range_stop_exclusive": stop,
        "note_rows_repaired": stop - start,
        "preset_rows_updated": preset_rows_updated,
        "handcrafted_dimensions": len(HANDCRAFTED_NAMES),
        "finite": bool(np.all(np.isfinite(features[start:stop]))),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("REPAIR_SUMMARY=" + json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

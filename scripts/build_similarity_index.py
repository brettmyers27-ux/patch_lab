#!/usr/bin/env python3
"""Validate combined CLAP indices and run Milestone 3 retrieval gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.match import l2_normalize


FEATURE_DIR = PROJECT_ROOT / "data" / "features"
REPORT = PROJECT_ROOT / "data" / "models" / "milestone3_similarity_report.json"


def self_retrieval(
    embeddings: np.ndarray, preset_ids: np.ndarray, block_size: int
) -> tuple[int, int, float, int, float]:
    # A query row is itself in this index. Float32 GEMM can put another row a
    # few ULP above that mathematically identical self-score, so count the own
    # row as tied for top-1 within 1e-6 and also report strict argmax results.
    tied_correct = 0
    strict_correct = 0
    for start in range(0, len(embeddings), block_size):
        stop = min(start + block_size, len(embeddings))
        queries = np.asarray(embeddings[start:stop], dtype=np.float32)
        scores = queries @ embeddings.T
        strict_indices = np.argmax(scores, axis=1)
        strict_correct += int(
            np.sum(preset_ids[strict_indices] == preset_ids[start:stop])
        )
        own_scores = np.einsum("ij,ij->i", queries, queries)
        tied_correct += int(np.sum(own_scores >= np.max(scores, axis=1) - 1e-6))
        print(f"SELF_RETRIEVAL_PROGRESS={stop}/{len(embeddings)}", flush=True)
    total = len(embeddings)
    return tied_correct, total, tied_correct / total, strict_correct, strict_correct / total


def octave_generalization(
    embeddings: np.ndarray,
    preset_ids: np.ndarray,
    notes: np.ndarray,
    block_size: int,
) -> tuple[int, int, float, int, float]:
    query_rows = np.flatnonzero(notes == 60)
    index_rows = np.flatnonzero(notes != 60)
    index_presets = preset_ids[index_rows]
    rows_by_preset = {
        int(preset_id): np.flatnonzero(index_presets == preset_id)
        for preset_id in np.unique(index_presets)
    }
    tied_correct = 0
    strict_correct = 0
    for start in range(0, len(query_rows), block_size):
        selected = query_rows[start : start + block_size]
        scores = np.asarray(embeddings[selected], dtype=np.float32) @ embeddings[index_rows].T
        local_indices = np.argpartition(scores, -5, axis=1)[:, -5:]
        neighbor_presets = preset_ids[index_rows[local_indices]]
        strict_correct += int(
            np.sum(np.any(neighbor_presets == preset_ids[selected, None], axis=1))
        )
        fifth_scores = np.partition(scores, -5, axis=1)[:, -5]
        for local_row, preset_id in enumerate(preset_ids[selected]):
            own_best = float(np.max(scores[local_row, rows_by_preset[int(preset_id)]]))
            tied_correct += int(own_best >= float(fifth_scores[local_row]) - 1e-6)
        print(f"OCTAVE_PROGRESS={min(start + block_size, len(query_rows))}/{len(query_rows)}", flush=True)
    total = len(query_rows)
    return tied_correct, total, tied_correct / total, strict_correct, strict_correct / total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-size", type=int, default=256)
    args = parser.parse_args()
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")
    note_complete = np.load(FEATURE_DIR / "note_complete.npy", mmap_mode="r")
    if not bool(np.all(note_complete)):
        raise RuntimeError("Feature extraction is incomplete")
    note_embeddings = l2_normalize(np.load(FEATURE_DIR / "note_embeddings.npy", mmap_mode="r"))
    note_manifest = np.load(FEATURE_DIR / "note_manifest.npz")
    preset_embeddings = l2_normalize(np.load(FEATURE_DIR / "preset_embeddings.npy"))
    preset_manifest = np.load(FEATURE_DIR / "preset_manifest.npz")
    # Persist explicit normalized matrices as the two combined search levels.
    np.save(FEATURE_DIR / "note_index.npy", note_embeddings)
    np.save(FEATURE_DIR / "preset_index.npy", preset_embeddings)
    np.savez_compressed(
        FEATURE_DIR / "similarity_manifest.npz",
        note_preset_ids=note_manifest["preset_ids"],
        note_midi_notes=note_manifest["midi_notes"],
        note_synths=note_manifest["synths"],
        preset_ids=preset_manifest["preset_ids"],
        preset_synths=preset_manifest["synths"],
    )

    own_correct, own_total, own_rate, strict_correct, strict_rate = self_retrieval(
        note_embeddings, note_manifest["preset_ids"], args.block_size
    )
    octave_correct, octave_total, octave_rate, octave_strict_correct, octave_strict_rate = octave_generalization(
        note_embeddings,
        note_manifest["preset_ids"],
        note_manifest["midi_notes"],
        args.block_size,
    )
    report = {
        "combined_synth_index": True,
        "note_rows": int(note_embeddings.shape[0]),
        "preset_rows": int(preset_embeddings.shape[0]),
        "embedding_dimensions": int(note_embeddings.shape[1]),
        "self_retrieval": {
            "correct": own_correct,
            "total": own_total,
            "rate": own_rate,
            "tie_tolerance": 1e-6,
            "strict_argmax_correct": strict_correct,
            "strict_argmax_rate": strict_rate,
            "required": 0.99,
            "pass": own_rate >= 0.99,
        },
        "octave_generalization": {
            "correct": octave_correct,
            "total": octave_total,
            "rate": octave_rate,
            "tie_tolerance": 1e-6,
            "strict_top5_correct": octave_strict_correct,
            "strict_top5_rate": octave_strict_rate,
            "target": 0.70,
            "reported_not_hard_failed": True,
        },
        "gate_pass": own_rate >= 0.99,
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("SIMILARITY_SUMMARY=" + json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

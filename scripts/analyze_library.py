#!/usr/bin/env python3
"""Resumably compute CLAP and handcrafted features for every render row."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.db import DEFAULT_DB_PATH
from core.features import (
    CLAP_CHECKPOINT,
    CLAP_DIMENSIONS,
    HANDCRAFTED_NAMES,
    ClapEmbedder,
    handcrafted_features,
    load_audio_48k_mono,
)
from core.platform_env import ENV


FEATURE_DIR = PROJECT_ROOT / "data" / "features"
NOTE_EMBEDDINGS = FEATURE_DIR / "note_embeddings.npy"
NOTE_FEATURES = FEATURE_DIR / "note_handcrafted.npy"
NOTE_COMPLETE = FEATURE_DIR / "note_complete.npy"
NOTE_MANIFEST = FEATURE_DIR / "note_manifest.npz"
PRESET_EMBEDDINGS = FEATURE_DIR / "preset_embeddings.npy"
PRESET_FEATURES = FEATURE_DIR / "preset_handcrafted.npy"
PRESET_MANIFEST = FEATURE_DIR / "preset_manifest.npz"
REPORT = PROJECT_ROOT / "data" / "models" / "milestone3_embedding_report.json"


def _rows(db_path: Path) -> list[sqlite3.Row]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection.execute(
        "SELECT r.preset_id,r.midi_note,r.wav_path,p.synth "
        "FROM renders r JOIN presets p ON p.id=r.preset_id "
        "ORDER BY r.preset_id,r.midi_note"
    ).fetchall()


def _open_array(path: Path, shape: tuple[int, ...], dtype: str) -> np.memmap:
    if path.exists():
        array = np.lib.format.open_memmap(path, mode="r+")
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise RuntimeError(f"Existing feature array has wrong shape/type: {path}")
        return array
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _write_manifest(rows: list[sqlite3.Row]) -> None:
    preset_ids = np.asarray([row["preset_id"] for row in rows], dtype=np.int64)
    notes = np.asarray([row["midi_note"] for row in rows], dtype=np.int16)
    synths = np.asarray([1 if row["synth"] == "serum1" else 2 for row in rows], dtype=np.uint8)
    paths = np.asarray([row["wav_path"] for row in rows])
    if NOTE_MANIFEST.exists():
        existing = np.load(NOTE_MANIFEST)
        if not np.array_equal(existing["preset_ids"], preset_ids) or not np.array_equal(
            existing["midi_notes"], notes
        ):
            raise RuntimeError("Render ordering changed since the feature manifest was created")
        return
    np.savez_compressed(
        NOTE_MANIFEST,
        preset_ids=preset_ids,
        midi_notes=notes,
        synths=synths,
        wav_paths=paths,
        handcrafted_names=np.asarray(HANDCRAFTED_NAMES),
    )


def _build_preset_level(
    rows: list[sqlite3.Row], embeddings: np.ndarray, features: np.ndarray
) -> tuple[int, float]:
    indices: dict[int, list[int]] = defaultdict(list)
    synth_by_preset: dict[int, int] = {}
    for index, row in enumerate(rows):
        preset_id = int(row["preset_id"])
        indices[preset_id].append(index)
        synth_by_preset[preset_id] = 1 if row["synth"] == "serum1" else 2
    preset_ids = np.asarray(sorted(indices), dtype=np.int64)
    preset_embeddings = np.empty((len(preset_ids), CLAP_DIMENSIONS), dtype=np.float32)
    preset_features = np.empty((len(preset_ids), len(HANDCRAFTED_NAMES)), dtype=np.float32)
    for row_index, preset_id in enumerate(preset_ids):
        selected = indices[int(preset_id)]
        mean_embedding = np.mean(embeddings[selected], axis=0)
        preset_embeddings[row_index] = mean_embedding / max(float(np.linalg.norm(mean_embedding)), 1e-12)
        preset_features[row_index] = np.mean(features[selected], axis=0)
    np.save(PRESET_EMBEDDINGS, preset_embeddings)
    np.save(PRESET_FEATURES, preset_features)
    np.savez_compressed(
        PRESET_MANIFEST,
        preset_ids=preset_ids,
        synths=np.asarray([synth_by_preset[int(preset_id)] for preset_id in preset_ids], dtype=np.uint8),
    )
    return len(preset_ids), float(np.max(np.abs(np.linalg.norm(preset_embeddings, axis=1) - 1.0)))


def main() -> int:
    global FEATURE_DIR, NOTE_EMBEDDINGS, NOTE_FEATURES, NOTE_COMPLETE
    global NOTE_MANIFEST, PRESET_EMBEDDINGS, PRESET_FEATURES, PRESET_MANIFEST
    global REPORT

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--feature-workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--expected-count", type=int, default=39_053)
    parser.add_argument("--feature-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.feature_workers <= 0:
        raise ValueError("--feature-workers must be positive")

    if args.feature_dir is not None:
        FEATURE_DIR = args.feature_dir.expanduser().resolve()
        NOTE_EMBEDDINGS = FEATURE_DIR / "note_embeddings.npy"
        NOTE_FEATURES = FEATURE_DIR / "note_handcrafted.npy"
        NOTE_COMPLETE = FEATURE_DIR / "note_complete.npy"
        NOTE_MANIFEST = FEATURE_DIR / "note_manifest.npz"
        PRESET_EMBEDDINGS = FEATURE_DIR / "preset_embeddings.npy"
        PRESET_FEATURES = FEATURE_DIR / "preset_handcrafted.npy"
        PRESET_MANIFEST = FEATURE_DIR / "preset_manifest.npz"
    if args.report is not None:
        REPORT = args.report.expanduser().resolve()

    rows = _rows(args.db)
    if len(rows) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count:,} render rows, found {len(rows):,}"
        )
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    _write_manifest(rows)
    embeddings = _open_array(NOTE_EMBEDDINGS, (len(rows), CLAP_DIMENSIONS), "float32")
    features = _open_array(NOTE_FEATURES, (len(rows), len(HANDCRAFTED_NAMES)), "float32")
    complete = _open_array(NOTE_COMPLETE, (len(rows),), "bool")
    pending = np.flatnonzero(~complete)
    if args.limit is not None:
        pending = pending[: args.limit]
    print(
        "ANALYZE_START="
        + json.dumps(
            {
                "backend": ENV.compute_backend,
                "checkpoint": CLAP_CHECKPOINT.name,
                "completed": int(complete.sum()),
                "pending_this_run": int(len(pending)),
                "total": len(rows),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if pending.size:
        embedder = ClapEmbedder(ENV)
    started = time.monotonic()
    processed = 0
    for offset in range(0, len(pending), args.batch_size):
        batch_indices = pending[offset : offset + args.batch_size]
        prepared = [load_audio_48k_mono(Path(rows[int(index)]["wav_path"])) for index in batch_indices]
        with ThreadPoolExecutor(max_workers=args.feature_workers) as pool:
            feature_futures = [
                pool.submit(handcrafted_features, item.waveform, item.sample_rate)
                for item in prepared
            ]
            # DSP and CLAP use different mixtures of FFT/convolution kernels;
            # overlap them on multi-core CPUs while keeping one CLAP instance.
            batch_embeddings = embedder.embed([item.waveform for item in prepared])
            batch_features = np.stack(
                [future.result() for future in feature_futures]
            )
        embeddings[batch_indices] = batch_embeddings
        features[batch_indices] = batch_features
        complete[batch_indices] = True
        embeddings.flush()
        features.flush()
        complete.flush()
        processed += len(batch_indices)
        elapsed = time.monotonic() - started
        rate = processed / elapsed if elapsed else 0.0
        remaining = len(pending) - processed
        print(
            "ANALYZE_PROGRESS="
            + json.dumps(
                {
                    "completed_total": int(complete.sum()),
                    "processed_this_run": processed,
                    "rate_per_second": rate,
                    "eta_seconds": remaining / rate if rate else None,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    all_complete = bool(np.all(complete))
    report = {
        "checkpoint": CLAP_CHECKPOINT.name,
        "backend": ENV.compute_backend,
        "render_count": len(rows),
        "completed": int(complete.sum()),
        "embedding_dimensions": CLAP_DIMENSIONS,
        "handcrafted_dimensions": len(HANDCRAFTED_NAMES),
        "handcrafted_names": list(HANDCRAFTED_NAMES),
        "note_embedding_max_norm_error": float(
            np.max(np.abs(np.linalg.norm(embeddings[complete], axis=1) - 1.0))
        )
        if np.any(complete)
        else None,
        "all_complete": all_complete,
    }
    if all_complete:
        preset_count, norm_error = _build_preset_level(rows, embeddings, features)
        report.update(
            {
                "preset_count": preset_count,
                "preset_embedding_max_norm_error": norm_error,
                "gate_pass": (
                    preset_count == 5_579
                    if args.expected_count == 39_053
                    else preset_count > 0
                ),
            }
        )
    else:
        report["gate_pass"] = False
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("ANALYZE_SUMMARY=" + json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

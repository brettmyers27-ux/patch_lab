#!/usr/bin/env python3
"""Generate, render, and embed resumable Serum 1 augmentation patches."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import librosa


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.dataset import _serum1_targets
from core.db import DEFAULT_DB_PATH
from core.features import CLAP_DIMENSIONS, HANDCRAFTED_NAMES, ClapEmbedder, handcrafted_features
from core.platform_env import ENV
from core.plugin_host import dump_dawdreamer_parameters, make_dawdreamer_processor
from core.synthetic import sample_serum1_patch


FEATURE_DIR = PROJECT_ROOT / "data" / "features"
REPORT = PROJECT_ROOT / "data" / "models" / "milestone3_synthetic_report.json"


def _array(path: Path, shape: tuple[int, ...], dtype: str) -> np.memmap:
    if path.exists():
        result = np.lib.format.open_memmap(path, mode="r+")
        if result.shape != shape or result.dtype != np.dtype(dtype):
            raise RuntimeError(f"Synthetic array has wrong shape/type: {path}")
        return result
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _render(engine, processor) -> np.ndarray:  # type: ignore[no-untyped-def]
    if hasattr(processor, "clear_midi"):
        processor.clear_midi()
    processor.add_midi_note(60, 100, 0.0, 2.5)
    engine.render(3.0)
    audio = np.asarray(engine.get_audio(), dtype=np.float32)
    if audio.shape[0] != 2 and audio.shape[1] == 2:
        audio = audio.T
    mono = np.ascontiguousarray(np.mean(audio, axis=0), dtype=np.float32)
    return librosa.resample(
        mono, orig_sr=44_100, target_sr=48_000, res_type="soxr_hq"
    ).astype(np.float32, copy=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    targets = _array(FEATURE_DIR / "synthetic_s1_targets.npy", (args.count, 316), "float32")
    embeddings = _array(
        FEATURE_DIR / "synthetic_s1_embeddings.npy", (args.count, CLAP_DIMENSIONS), "float32"
    )
    features = _array(
        FEATURE_DIR / "synthetic_s1_handcrafted.npy",
        (args.count, len(HANDCRAFTED_NAMES)),
        "float32",
    )
    complete = _array(FEATURE_DIR / "synthetic_s1_complete.npy", (args.count,), "bool")
    store = _serum1_targets(DEFAULT_DB_PATH)
    candidate = next(
        item for item in ENV.plugins_for("serum1") if item.format == "VST2" and item.hostable
    )
    engine, processor = make_dawdreamer_processor(candidate)
    initial = np.asarray(
        [parameter.norm_value for parameter in dump_dawdreamer_parameters(processor)],
        dtype=np.float32,
    )
    if initial.shape != (316,):
        raise RuntimeError(f"Expected 316 Serum 1 parameters, received {initial.shape}")
    embedder = ClapEmbedder(ENV)
    pending = np.flatnonzero(~complete)
    started = time.monotonic()
    processed = 0
    silent_retries = 0
    for offset in range(0, len(pending), args.batch_size):
        indices = pending[offset : offset + args.batch_size]
        waveforms = []
        for index in indices:
            # An index-local seed makes interrupted/resumed output bit-for-bit
            # independent of which earlier rows needed an inaudible retry.
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, int(index)]))
            for attempt in range(16):
                vector = sample_serum1_patch(rng, initial, store.mapping)
                for parameter_index, value in enumerate(vector):
                    processor.set_parameter(parameter_index, float(value))
                waveform = _render(engine, processor)
                rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
                if rms > 1e-3:
                    break
                silent_retries += 1
            else:
                raise RuntimeError(
                    f"Could not produce an audible synthetic patch for row {int(index)} "
                    "after 16 deterministic attempts"
                )
            targets[int(index)] = vector
            waveforms.append(waveform)
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(handcrafted_features, waveform) for waveform in waveforms]
            embeddings[indices] = embedder.embed(waveforms)
            features[indices] = np.stack([future.result() for future in futures])
        complete[indices] = True
        for array in (targets, embeddings, features, complete):
            array.flush()
        processed += len(indices)
        elapsed = time.monotonic() - started
        rate = processed / elapsed if elapsed else 0.0
        print(
            "SYNTHETIC_PROGRESS="
            + json.dumps(
                {
                    "complete": int(np.sum(complete)),
                    "total": args.count,
                    "rate_per_second": rate,
                    "eta_seconds": (len(pending) - processed) / rate if rate else None,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    report = {
        "requested": args.count,
        "completed": int(np.sum(complete)),
        "silent_retries_this_run": silent_retries,
        "silent_examples_stored": 0,
        "seed": args.seed,
        "mod_matrix_kept_at_init": True,
        "audible_source_priors": True,
        "gate_pass": bool(np.all(complete)),
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("SYNTHETIC_SUMMARY=" + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

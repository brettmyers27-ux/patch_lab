#!/usr/bin/env python3
"""Generate resumable on-manifold Serum 1 and Serum 2 training patches."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sqlite3
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from core.dataset import FEATURE_DIR, _serum1_targets, _serum2_targets
from core.db import DEFAULT_DB_PATH
from core.features import CLAP_DIMENSIONS, HANDCRAFTED_NAMES, ClapEmbedder, handcrafted_features
from core.perturbation import perturb_serum1, perturb_serum2
from core.platform_env import ENV


REPORT = PROJECT_ROOT / "data" / "models" / "milestone3_perturbation_report.json"
SPOT_ROOT = PROJECT_ROOT / "data" / "models" / "perturbation_spot"
STATE_ROOT = PROJECT_ROOT / "data" / "models" / "serum2_render_states"
SAMPLE_RATE = 44_100
TARGET_RATE = 48_000
_WORKER: dict[str, Any] = {}


def _array(path: Path, shape: tuple[int, ...], dtype: str) -> np.memmap:
    if path.exists():
        result = np.lib.format.open_memmap(path, mode="r+")
        if result.shape != shape or result.dtype != np.dtype(dtype):
            raise RuntimeError(f"Perturbation array has wrong shape/type: {path}")
        return result
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _eligible(synth: str, store: Any) -> tuple[np.ndarray, dict[int, str]]:
    connection = sqlite3.connect(DEFAULT_DB_PATH)
    rows = connection.execute(
        "SELECT p.id,p.path FROM presets p JOIN renders r ON r.preset_id=p.id "
        "WHERE p.synth=? AND p.status='rendered' AND r.midi_note=60 AND r.rms_dbfs>-60 "
        "ORDER BY p.id",
        (synth,),
    ).fetchall()
    paths = {int(preset_id): str(path) for preset_id, path in rows}
    ids = np.asarray(
        [preset_id for preset_id in paths if preset_id in store.preset_row], dtype=np.int64
    )
    if ids.size == 0:
        raise RuntimeError(f"No audible real bases for {synth}")
    return ids, paths


def _init_worker(synth: str, seed: int) -> None:
    from core.plugin_host import make_dawdreamer_processor

    store = _serum1_targets(DEFAULT_DB_PATH) if synth == "serum1" else _serum2_targets()
    ids, paths = _eligible(synth, store)
    required = "VST2" if synth == "serum1" else "VST3"
    candidate = next(
        item for item in ENV.plugins_for(synth) if item.format == required and item.hostable
    )
    engine, processor = make_dawdreamer_processor(candidate)
    state_path = Path(tempfile.mkdtemp(prefix=f"patchlab-{synth}-{os.getpid()}-")) / "state.vstpreset"
    schema = None
    if synth == "serum2":
        schema = json.loads(
            (PROJECT_ROOT / "data" / "models" / "serum2_target_schema.json").read_text()
        )
    _WORKER.update(
        synth=synth,
        seed=seed,
        store=store,
        ids=ids,
        paths=paths,
        engine=engine,
        processor=processor,
        schema=schema,
        state_path=state_path,
    )


def _render() -> np.ndarray:
    engine, processor = _WORKER["engine"], _WORKER["processor"]
    if hasattr(processor, "clear_midi"):
        processor.clear_midi()
    processor.add_midi_note(60, 100, 0.0, 2.5)
    engine.render(3.0)
    audio = np.asarray(engine.get_audio(), dtype=np.float32)
    if audio.shape[0] != 2 and audio.shape[1] == 2:
        audio = audio.T
    mono = np.ascontiguousarray(np.mean(audio, axis=0), dtype=np.float32)
    return librosa.resample(
        mono, orig_sr=SAMPLE_RATE, target_sr=TARGET_RATE, res_type="soxr_hq"
    ).astype(np.float32, copy=False)


def _apply_serum2(vector: np.ndarray, mask: np.ndarray, preset_id: int) -> float:
    from core.serum2_preset import Serum2Preset
    from core.serum2_state_reconstruct import (
        decode_host_template,
        reconstruct_partial_vstpreset,
    )
    from core.serum2_targets import decode_vector

    template = decode_host_template((STATE_ROOT / f"{preset_id}.vstpreset").read_bytes())
    graph = decode_vector(vector, _WORKER["schema"], mask)
    predicted = Serum2Preset(
        path=Path(_WORKER["paths"][preset_id]),
        metadata={"presetName": f"Perturbation {preset_id}"},
        data=graph,
        metadata_length=0,
        cbor_length=0,
        payload_version=0,
        compressed_length=0,
    )
    container, partition = reconstruct_partial_vstpreset(
        predicted, template, merge_matching_lists=True
    )
    state_path: Path = _WORKER["state_path"]
    state_path.write_bytes(container)
    if _WORKER["processor"].load_vst3_preset(str(state_path)) is False:
        raise RuntimeError("Serum 2 rejected perturbed state")
    return partition.coverage


def _work(index: int) -> dict[str, Any]:
    synth = str(_WORKER["synth"])
    store = _WORKER["store"]
    discards = 0
    last_error = ""
    for attempt in range(16):
        rng = np.random.default_rng(
            np.random.SeedSequence([int(_WORKER["seed"]), 1 if synth == "serum1" else 2, index, attempt])
        )
        preset_id = int(rng.choice(_WORKER["ids"]))
        row = store.preset_row[preset_id]
        base, mask = store.vectors[row], store.masks[row]
        try:
            if synth == "serum1":
                vector, change = perturb_serum1(base, mask, store.mapping, rng)
                if _WORKER["processor"].load_preset(_WORKER["paths"][preset_id]) is False:
                    raise RuntimeError("Serum 1 rejected base preset")
                for parameter_index, value in enumerate(vector):
                    _WORKER["processor"].set_parameter(parameter_index, float(value))
                coverage = 1.0
            else:
                vector, change = perturb_serum2(base, mask, _WORKER["schema"], rng)
                coverage = _apply_serum2(vector, mask, preset_id)
            waveform = _render()
            rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
            if not np.all(np.isfinite(waveform)) or rms <= 1e-3:
                raise RuntimeError(f"inaudible/non-finite render rms={rms}")
            return {
                "index": index,
                "preset_id": preset_id,
                "vector": vector,
                "waveform": waveform,
                "rms_dbfs": float(20.0 * np.log10(max(rms, 1e-12))),
                "discards": discards,
                "coverage": coverage,
                "change": change,
            }
        except Exception as exc:
            discards += 1
            last_error = f"{type(exc).__name__}: {exc}"
    return {"index": index, "error": last_error, "traceback": traceback.format_exc()}


def _real_distribution(synth: str) -> dict[str, list[float]]:
    manifest = np.load(FEATURE_DIR / "note_manifest.npz")
    synth_code = 1 if synth == "serum1" else 2
    select = (manifest["synths"] == synth_code) & (manifest["midi_notes"] == 60)
    features = np.load(FEATURE_DIR / "note_handcrafted.npy", mmap_mode="r")[select]
    connection = sqlite3.connect(DEFAULT_DB_PATH)
    rms = np.asarray(
        [
            float(row[0])
            for row in connection.execute(
                "SELECT r.rms_dbfs FROM renders r JOIN presets p ON p.id=r.preset_id "
                "WHERE p.synth=? AND r.midi_note=60 AND r.rms_dbfs>-60",
                (synth,),
            )
        ],
        dtype=np.float64,
    )
    return {
        "rms_dbfs": np.percentile(rms, [5, 50, 95]).tolist(),
        "centroid_hz": np.percentile(features[:, 0], [5, 50, 95]).tolist(),
    }


def _generate(
    synth: str,
    count: int,
    processes: int,
    batch_size: int,
    seed: int,
    limit: int | None,
) -> dict[str, Any]:
    store = _serum1_targets(DEFAULT_DB_PATH) if synth == "serum1" else _serum2_targets()
    tag = "s1" if synth == "serum1" else "s2"
    targets = _array(FEATURE_DIR / f"perturb_{tag}_targets.npy", (count, store.dimension), "float32")
    embeddings = _array(
        FEATURE_DIR / f"perturb_{tag}_embeddings.npy", (count, CLAP_DIMENSIONS), "float32"
    )
    features = _array(
        FEATURE_DIR / f"perturb_{tag}_handcrafted.npy", (count, len(HANDCRAFTED_NAMES)), "float32"
    )
    base_ids = _array(FEATURE_DIR / f"perturb_{tag}_base_ids.npy", (count,), "int64")
    rms_values = _array(FEATURE_DIR / f"perturb_{tag}_rms_dbfs.npy", (count,), "float32")
    discard_counts = _array(FEATURE_DIR / f"perturb_{tag}_discards.npy", (count,), "uint8")
    enum_changed = _array(FEATURE_DIR / f"perturb_{tag}_enum_changed.npy", (count,), "bool")
    coverage_values = _array(FEATURE_DIR / f"perturb_{tag}_coverage.npy", (count,), "float32")
    complete = _array(FEATURE_DIR / f"perturb_{tag}_complete.npy", (count,), "bool")
    pending = list(map(int, np.flatnonzero(~complete)))
    if limit is not None:
        pending = pending[:limit]
    embedder = ClapEmbedder(ENV)
    context = mp.get_context("spawn")
    started = time.monotonic()
    processed = discards = enum_changes = 0
    coverages: list[float] = []
    spot_dir = SPOT_ROOT / synth
    spot_dir.mkdir(parents=True, exist_ok=True)
    with context.Pool(processes, initializer=_init_worker, initargs=(synth, seed)) as pool:
        iterator = pool.imap_unordered(_work, pending, chunksize=1)
        batch: list[dict[str, Any]] = []
        for result in iterator:
            if "error" in result:
                raise RuntimeError(f"Perturbation {synth} row {result['index']} failed: {result['error']}")
            batch.append(result)
            if len(batch) < batch_size and processed + len(batch) < len(pending):
                continue
            waveforms = [row["waveform"] for row in batch]
            with ThreadPoolExecutor(max_workers=8) as threads:
                feature_jobs = [threads.submit(handcrafted_features, waveform) for waveform in waveforms]
                batch_embeddings = embedder.embed(waveforms)
                batch_features = np.stack([job.result() for job in feature_jobs])
            for position, row in enumerate(batch):
                index = int(row["index"])
                targets[index] = row["vector"]
                embeddings[index] = batch_embeddings[position]
                features[index] = batch_features[position]
                base_ids[index] = int(row["preset_id"])
                rms_values[index] = float(row["rms_dbfs"])
                discard_counts[index] = int(row["discards"])
                enum_changed[index] = bool(row["change"]["enum_changed"])
                coverage_values[index] = float(row["coverage"])
                complete[index] = True
                discards += int(row["discards"])
                enum_changes += int(row["change"]["enum_changed"])
                coverages.append(float(row["coverage"]))
                if index < 20:
                    sf.write(spot_dir / f"{index}.wav", row["waveform"], TARGET_RATE, subtype="FLOAT")
            for array in (
                targets,
                embeddings,
                features,
                base_ids,
                rms_values,
                discard_counts,
                enum_changed,
                coverage_values,
                complete,
            ):
                array.flush()
            processed += len(batch)
            elapsed = max(time.monotonic() - started, 1e-6)
            rate = processed / elapsed
            print(
                "PERTURB_PROGRESS="
                + json.dumps(
                    {
                        "synth": synth,
                        "complete": int(np.sum(complete)),
                        "total": count,
                        "rate_per_second": rate,
                        "eta_seconds": (len(pending) - processed) / rate if rate else None,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            batch.clear()
    selected_rows = np.asarray(complete, dtype=np.bool_)
    stored_discards = int(np.sum(np.asarray(discard_counts)[selected_rows]))
    stored_count = int(np.sum(selected_rows))
    attempts = stored_count + stored_discards
    synth_features = np.asarray(features)[selected_rows]
    synthetic_distribution = {
        "rms_dbfs": np.percentile(np.asarray(rms_values)[selected_rows], [5, 50, 95]).tolist(),
        "centroid_hz": np.percentile(synth_features[:, 0], [5, 50, 95]).tolist(),
    }
    real_distribution = _real_distribution(synth)
    centroid_ratio = synthetic_distribution["centroid_hz"][1] / max(
        real_distribution["centroid_hz"][1], 1e-9
    )
    plausible = 0.25 <= centroid_ratio <= 4.0 and abs(
        synthetic_distribution["rms_dbfs"][1] - real_distribution["rms_dbfs"][1]
    ) <= 20.0
    return {
        "requested": count,
        "completed": stored_count,
        "new_this_run": processed,
        "discarded_attempts": stored_discards,
        "discard_rate": stored_discards / attempts if attempts else 0.0,
        "enum_change_rate": float(np.mean(np.asarray(enum_changed)[selected_rows])),
        "mean_structural_coverage": float(np.mean(np.asarray(coverage_values)[selected_rows])),
        "real_distribution_p05_p50_p95": real_distribution,
        "perturbation_distribution_p05_p50_p95": synthetic_distribution,
        "spot_render_count": len(list(spot_dir.glob("*.wav"))),
        "plausible_distribution": plausible,
        "gate_pass": bool(np.all(complete))
        and (stored_discards / attempts if attempts else 0.0) <= 0.15
        and plausible,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth", choices=("serum1", "serum2", "both"), default="both")
    parser.add_argument("--count", type=int, default=20_000)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int, help="Process at most this many pending rows per synth")
    args = parser.parse_args()
    selected = ("serum1", "serum2") if args.synth == "both" else (args.synth,)
    existing = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    for synth in selected:
        existing[synth] = _generate(
            synth, args.count, args.processes, args.batch_size, args.seed, args.limit
        )
        REPORT.write_text(json.dumps(existing, indent=2, sort_keys=True))
        print("PERTURB_SUMMARY=" + json.dumps({synth: existing[synth]}, sort_keys=True))
        if args.limit is None and not existing[synth]["gate_pass"]:
            return 1
    existing["gate_pass"] = all(existing.get(synth, {}).get("gate_pass") for synth in ("serum1", "serum2"))
    REPORT.write_text(json.dumps(existing, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

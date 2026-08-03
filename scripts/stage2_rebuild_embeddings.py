#!/usr/bin/env python3
"""Preflight and atomically rebuild the complete Stage 2 embedding world."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from core.dataset import _serum1_targets, _serum2_targets
from core.factory_bundle import DEFAULT_FACTORY_BUNDLE
from core.factory_verify import verify_local_factory_install
from core.features import CLAP_DIMENSIONS, HANDCRAFTED_NAMES, ClapEmbedder, handcrafted_features
from core.matcher import Candidate, _init_render_worker, _render_candidate
from core.model_assets import configure_model_environment
from core.platform_env import ENV
from core.synthesis_assets import resolve_synthesis_assets


DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "stage2" / "artifacts-v2"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "stage2" / "embedding-rebuild-report.json"


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def renderability_inventory(
    preset_ids: Sequence[int], synths: Sequence[int], *, library_db: Path
) -> dict[str, Any]:
    assets = resolve_synthesis_assets()
    # ``factory_mapping`` is an input to the render worker.  It may be the
    # Stage 2B transferred-preset map, so never pass it as
    # ``verify_local_factory_install(mapping_path=...)``: that argument is an
    # output path and would overwrite the map with a scanner-only result.
    verification = verify_local_factory_install()
    local_paths = dict(verification.local_paths_by_hash)
    if assets.factory_mapping is not None and Path(assets.factory_mapping).is_file():
        try:
            payload = json.loads(Path(assets.factory_mapping).read_text(encoding="utf-8"))
            for content_hash, local_path in payload.get("local_paths_by_hash", {}).items():
                if Path(str(local_path)).is_file():
                    local_paths[str(content_hash)] = str(local_path)
        except (OSError, ValueError, TypeError):
            pass
    with closing(sqlite3.connect(library_db)) as connection:
        catalog = {
            int(preset_id): (str(path or ""), str(content_hash or ""))
            for preset_id, path, content_hash in connection.execute(
                "SELECT id,path,content_hash FROM presets"
            )
        }
    available: list[int] = []
    missing: list[dict[str, Any]] = []
    by_synth = {
        "serum1": {"required": 0, "available": 0, "missing": 0},
        "serum2": {"required": 0, "available": 0, "missing": 0},
    }
    for preset_id_value, synth_code_value in zip(preset_ids, synths, strict=True):
        preset_id = int(preset_id_value)
        code = int(synth_code_value)
        name = "serum1" if code == 1 else "serum2"
        by_synth[name]["required"] += 1
        row = catalog.get(preset_id)
        if row is None:
            reason = "not present in synthesis catalog"
        elif code == 1:
            stored_path, content_hash = row
            local = local_paths.get(content_hash, "")
            reason = "" if Path(stored_path).is_file() or (local and Path(local).is_file()) else "Serum 1 preset file absent on this PC"
        else:
            reason = "" if assets.find_render_state(preset_id) is not None else "Serum 2 render-state template absent"
        if reason:
            by_synth[name]["missing"] += 1
            missing.append({"preset_id": preset_id, "synth": name, "reason": reason})
        else:
            by_synth[name]["available"] += 1
            available.append(preset_id)
    return {
        "required": len(preset_ids),
        "available": len(available),
        "missing": len(missing),
        "by_synth": by_synth,
        "missing_presets": missing,
        "complete_embedding_world_possible": not missing,
    }


def _open_array(path: Path, shape: tuple[int, ...], dtype: str) -> np.memmap:
    if path.is_file():
        array = np.lib.format.open_memmap(path, mode="r+")
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise RuntimeError(f"Existing rebuild array has wrong shape/type: {path}")
        return array
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _copy_runtime_targets(source_features: Path, output: Path) -> None:
    """Carry forward non-embedding inputs required by the v2 predictor runtime."""

    source = source_features / "serum2_targets.npz"
    if not source.is_file():
        raise RuntimeError(f"Missing Serum 2 runtime targets in source features: {source}")
    destination = output / source.name
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _close_memmaps(*arrays: np.memmap) -> None:
    """Release Windows file handles before child tools replace index files."""

    for array in arrays:
        array.flush()
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            mapping.close()


def _build_preset_level(
    output: Path,
    preset_ids: np.ndarray,
    synths: np.ndarray,
    note_embeddings: np.ndarray,
    note_features: np.ndarray,
) -> None:
    rows: dict[int, list[int]] = defaultdict(list)
    synth_by_preset: dict[int, int] = {}
    for row, (preset_id, synth) in enumerate(zip(preset_ids, synths, strict=True)):
        rows[int(preset_id)].append(row)
        synth_by_preset[int(preset_id)] = int(synth)
    unique = np.asarray(sorted(rows), dtype=np.int64)
    embeddings = np.empty((len(unique), CLAP_DIMENSIONS), dtype=np.float32)
    features = np.empty((len(unique), len(HANDCRAFTED_NAMES)), dtype=np.float32)
    for row, preset_id in enumerate(unique):
        selected = rows[int(preset_id)]
        mean = np.mean(note_embeddings[selected], axis=0)
        embeddings[row] = mean / max(float(np.linalg.norm(mean)), 1e-12)
        features[row] = np.mean(note_features[selected], axis=0)
    np.save(output / "preset_embeddings.npy", embeddings)
    np.save(output / "preset_handcrafted.npy", features)
    np.savez_compressed(
        output / "preset_manifest.npz",
        preset_ids=unique,
        synths=np.asarray([synth_by_preset[int(value)] for value in unique], dtype=np.uint8),
    )


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-feature-dir", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--source-bundle", type=Path, default=DEFAULT_FACTORY_BUNDLE)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.processes != 4:
        raise ValueError("Stage 2 embedding rebuild uses the verified 4-worker render pool")
    os.environ.setdefault("PATCHLAB_DISTRIBUTION_MODE", "1")
    os.environ["PATCHLAB_CLAP_CHECKPOINT"] = str(args.checkpoint.expanduser().resolve())
    configure_model_environment()
    assets = resolve_synthesis_assets()
    source_features = (args.source_feature_dir or assets.feature_dir).expanduser().resolve()
    old_manifest = np.load(source_features / "similarity_manifest.npz")
    preset_ids = np.asarray(old_manifest["note_preset_ids"], dtype=np.int64)
    midi_notes = np.asarray(old_manifest["note_midi_notes"], dtype=np.int16)
    synths = np.asarray(old_manifest["note_synths"], dtype=np.uint8)
    unique_ids = np.asarray(old_manifest["preset_ids"], dtype=np.int64)
    unique_synths = np.asarray(old_manifest["preset_synths"], dtype=np.uint8)
    inventory = renderability_inventory(unique_ids, unique_synths, library_db=assets.library_db)
    report: dict[str, Any] = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "source_feature_dir": str(source_features),
        "renderability": inventory,
        "required_note_rows": len(preset_ids),
    }
    report_path = args.report.expanduser().resolve()
    if args.preflight_only or not inventory["complete_embedding_world_possible"]:
        report["status"] = "ready" if not inventory["missing"] else "blocked-incomplete-source-library"
        report["reason"] = (
            None
            if not inventory["missing"]
            else "A partial rebuild would mix encoder worlds or shrink retrieval scope, so no v2 index was created."
        )
        _atomic_json(report_path, report)
        print("STAGE2_EMBEDDING_PREFLIGHT=" + json.dumps({key: value for key, value in report.items() if key != "renderability"}, sort_keys=True))
        return 0 if args.preflight_only else 2

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _copy_runtime_targets(source_features, output)
    np.savez_compressed(
        output / "note_manifest.npz",
        preset_ids=preset_ids,
        midi_notes=midi_notes,
        synths=synths,
        handcrafted_names=np.asarray(HANDCRAFTED_NAMES),
    )
    note_embeddings = _open_array(output / "note_embeddings.npy", (len(preset_ids), CLAP_DIMENSIONS), "float32")
    note_features = _open_array(output / "note_handcrafted.npy", (len(preset_ids), len(HANDCRAFTED_NAMES)), "float32")
    complete = _open_array(output / "note_complete.npy", (len(preset_ids),), "bool")
    stores = {1: _serum1_targets(assets.library_db), 2: _serum2_targets(assets.serum2_targets, assets.serum2_schema)}
    pending = np.flatnonzero(~complete)
    embedder = ClapEmbedder(ENV, checkpoint=args.checkpoint.expanduser().resolve())
    context = mp.get_context("spawn")
    started = time.monotonic()
    scratch = tempfile.TemporaryDirectory(prefix="patchlab-stage2-index-")
    try:
        with context.Pool(args.processes, initializer=_init_render_worker, initargs=(scratch.name, assets)) as pool:
            for offset in range(0, len(pending), args.batch_size):
                rows = pending[offset : offset + args.batch_size]
                candidates = []
                for row in rows:
                    code = int(synths[row])
                    preset_id = int(preset_ids[row])
                    store = stores[code]
                    target_row = store.preset_row[preset_id]
                    candidates.append(
                        Candidate(
                            "serum1" if code == 1 else "serum2",
                            preset_id,
                            np.asarray(store.vectors[target_row], dtype=np.float32),
                            np.asarray(store.masks[target_row], dtype=bool),
                            "stage2-index-rebuild",
                            exact_base=True,
                            midi_note=int(midi_notes[row]),
                        )
                    )
                results = pool.map(
                    _render_candidate,
                    [(candidate, int(midi_notes[row]), 4.0) for candidate, row in zip(candidates, rows, strict=True)],
                    chunksize=1,
                )
                errors = [(int(row), error) for row, (_audio, _coverage, error) in zip(rows, results, strict=True) if error]
                if errors:
                    raise RuntimeError(f"Embedding rebuild render failed: first={errors[0]}")
                audio = [np.asarray(result[0], dtype=np.float32) for result in results]
                embedded = []
                for start in range(0, len(audio), args.embedding_batch_size):
                    embedded.append(embedder.embed(audio[start : start + args.embedding_batch_size]))
                with ThreadPoolExecutor(max_workers=min(8, len(audio))) as executor:
                    handcrafted = np.stack(list(executor.map(handcrafted_features, audio)))
                note_embeddings[rows] = np.concatenate(embedded, axis=0)
                note_features[rows] = handcrafted
                complete[rows] = True
                note_embeddings.flush(); note_features.flush(); complete.flush()
                elapsed = time.monotonic() - started
                processed = offset + len(rows)
                print("STAGE2_INDEX_RENDER_PROGRESS=" + json.dumps({"processed_this_run": processed, "pending_this_run": len(pending), "rate": processed / max(elapsed, 1e-9)}, sort_keys=True), flush=True)
    finally:
        scratch.cleanup()
    _build_preset_level(output, preset_ids, synths, note_embeddings, note_features)
    # NumPy keeps memmap file handles open.  On Windows a child process cannot
    # atomically replace an index file while its source mmap is still open.
    _close_memmaps(note_embeddings, note_features, complete)
    python = str(Path(os.sys.executable).resolve())
    _run([python, "scripts/build_similarity_index.py", "--feature-dir", str(output), "--report", str(output / "similarity-report.json")])
    _run([python, "scripts/build_delta_neighbors.py", "--feature-dir", str(output), "--output", str(output / "delta_neighbors.npz"), "--report", str(output / "delta-neighbor-report.json")])
    _run([
        python, "scripts/build_factory_bundle.py", "--source-bundle", str(args.source_bundle.expanduser().resolve()),
        "--catalog", str(assets.library_db), "--feature-dir", str(output),
        "--output", str(output / "factory_bundle.sqlite"), "--report", str(output / "factory-bundle-report.json"),
    ])
    report.update({
        "status": "complete",
        "note_rows": len(preset_ids),
        "preset_rows": len(unique_ids),
        "elapsed_s": time.monotonic() - started,
        "output": str(output),
    })
    _atomic_json(report_path, report)
    print("STAGE2_EMBEDDING_REBUILD=" + json.dumps({key: value for key, value in report.items() if key != "renderability"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

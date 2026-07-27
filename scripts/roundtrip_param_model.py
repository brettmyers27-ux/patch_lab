#!/usr/bin/env python3
"""Render 20 validation predictions per synth and score CLAP fidelity."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import librosa
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.db import DEFAULT_DB_PATH
from core.features import ClapEmbedder, handcrafted_features
from core.platform_env import ENV
from core.plugin_host import make_dawdreamer_processor
from core.render import _trim_tail
from core.serum2_preset import Serum2Preset
from core.serum2_state_reconstruct import decode_host_template, reconstruct_partial_vstpreset
from core.serum2_targets import decode_vector
from core.train import load_parameter_model, predict_parameters


FEATURE_DIR = PROJECT_ROOT / "data" / "features"
STATE_DIR = PROJECT_ROOT / "data" / "models" / "roundtrip_serum2_states"
REPORT = PROJECT_ROOT / "data" / "models" / "milestone3_roundtrip_report.json"


def _render(engine: Any, processor: Any) -> np.ndarray:
    if hasattr(processor, "clear_midi"):
        processor.clear_midi()
    processor.add_midi_note(60, 100, 0.0, 4.0)
    engine.render(8.0)
    audio = np.asarray(engine.get_audio(), dtype=np.float32)
    if audio.shape[0] != 2 and audio.shape[1] == 2:
        audio = audio.T
    audio = _trim_tail(audio)
    mono = np.mean(audio, axis=0, dtype=np.float32)
    return librosa.resample(
        mono, orig_sr=44_100, target_sr=48_000, res_type="soxr_hq"
    ).astype(np.float32, copy=False)


def _feature_rows() -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], int]]:
    embeddings = np.load(FEATURE_DIR / "note_embeddings.npy", mmap_mode="r")
    handcrafted = np.load(FEATURE_DIR / "note_handcrafted.npy", mmap_mode="r")
    manifest = np.load(FEATURE_DIR / "note_manifest.npz")
    lookup = {
        (int(preset_id), int(note)): index
        for index, (preset_id, note) in enumerate(
            zip(manifest["preset_ids"], manifest["midi_notes"], strict=True)
        )
    }
    return embeddings, handcrafted, lookup


def main() -> int:
    model, checkpoint = load_parameter_model(device=ENV.compute_backend)
    rng = np.random.default_rng(int(checkpoint["split"]["seed"]))
    selected: dict[str, list[int]] = {}
    for synth in ("serum1", "serum2"):
        candidates = np.asarray(checkpoint["split"]["validation_preset_ids"][synth], dtype=np.int64)
        selected[synth] = list(map(int, rng.choice(candidates, size=20, replace=False)))
    embeddings, features, lookup = _feature_rows()
    connection = sqlite3.connect(DEFAULT_DB_PATH)
    connection.row_factory = sqlite3.Row

    s1_candidate = next(
        item for item in ENV.plugins_for("serum1") if item.format == "VST2" and item.hostable
    )
    s2_candidate = next(
        item for item in ENV.plugins_for("serum2") if item.format == "VST3" and item.hostable
    )
    s1_engine, s1_processor = make_dawdreamer_processor(s1_candidate)
    s2_engine, s2_processor = make_dawdreamer_processor(s2_candidate)
    from pedalboard import load_plugin

    live = load_plugin(str(s2_candidate.path), plugin_name="Serum 2")
    template = decode_host_template(bytes(live.preset_data))
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    rendered: list[np.ndarray] = []
    originals: list[np.ndarray] = []
    details: list[dict[str, Any]] = []

    for synth in ("serum1", "serum2"):
        for preset_id in selected[synth]:
            row_index = lookup[(preset_id, 60)]
            prediction = predict_parameters(
                model,
                checkpoint,
                embeddings[row_index],
                features[row_index],
                synth,
            )
            detail: dict[str, Any] = {"preset_id": preset_id, "synth": synth}
            if synth == "serum1":
                for parameter_index, value in enumerate(prediction):
                    s1_processor.set_parameter(parameter_index, float(value))
                audio = _render(s1_engine, s1_processor)
            else:
                settings = decode_vector(prediction, checkpoint["serum2_schema"])
                source = connection.execute(
                    "SELECT p.path,f.metadata_json,f.payload_version,f.cbor_length,f.compressed_length "
                    "FROM presets p JOIN serum2_full_settings f ON f.preset_id=p.id WHERE p.id=?",
                    (preset_id,),
                ).fetchone()
                decoded = Serum2Preset(
                    path=Path(source["path"]),
                    metadata=json.loads(source["metadata_json"]),
                    data=settings,
                    metadata_length=len(source["metadata_json"].encode("utf-8")),
                    cbor_length=int(source["cbor_length"]),
                    payload_version=int(source["payload_version"]),
                    compressed_length=int(source["compressed_length"]),
                )
                container, partition = reconstruct_partial_vstpreset(decoded, template)
                state_path = STATE_DIR / f"{preset_id}.vstpreset"
                state_path.write_bytes(container)
                if s2_processor.load_vst3_preset(str(state_path)) is False:
                    raise RuntimeError(f"Serum 2 rejected predicted state {state_path}")
                detail["structural_coverage"] = partition.coverage
                detail["state_path"] = str(state_path)
                audio = _render(s2_engine, s2_processor)
            rendered.append(audio)
            originals.append(np.asarray(embeddings[row_index], dtype=np.float32))
            detail["render_rms_dbfs"] = float(
                20.0 * np.log10(max(float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))), 1e-12))
            )
            details.append(detail)
            print(f"ROUNDTRIP_RENDER={len(details)}/40", flush=True)

    embedder = ClapEmbedder(ENV)
    predicted_embeddings = embedder.embed(rendered)
    for detail, original, predicted in zip(details, originals, predicted_embeddings, strict=True):
        detail["clap_cosine_similarity"] = float(np.dot(original, predicted))
    by_synth = {}
    for synth in ("serum1", "serum2"):
        rows = [row for row in details if row["synth"] == synth]
        by_synth[synth] = {
            "count": len(rows),
            "mean_clap_cosine_similarity": float(
                np.mean([row["clap_cosine_similarity"] for row in rows])
            ),
            "minimum_clap_cosine_similarity": float(
                np.min([row["clap_cosine_similarity"] for row in rows])
            ),
            "silent_renders": sum(row["render_rms_dbfs"] <= -60.0 for row in rows),
        }
        coverages = [row["structural_coverage"] for row in rows if "structural_coverage" in row]
        if coverages:
            by_synth[synth]["mean_partial_overlay_coverage"] = float(np.mean(coverages))
    report = {
        "by_synth": by_synth,
        "presets": details,
        "serum2_reconstruction": (
            "Sparse predicted setting leaves are overlaid on the valid live init state; "
            "omitted internal structure and unsupported variable topology remain at init."
        ),
        "gate_pass": len(details) == 40
        and all(result["silent_renders"] == 0 for result in by_synth.values()),
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("ROUNDTRIP_SUMMARY=" + json.dumps(by_synth, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate and audio-verify ten native preset exports across both synths."""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.dataset import _serum1_targets, _serum2_targets
from core.db import DEFAULT_DB_PATH, Database
from core.matcher import Candidate
from core.perturbation import perturb_serum1, perturb_serum2
from core.preset_export import PresetExportVerifier, write_and_verify_native_preset
from core.serum2_preset import Serum2Preset
from core.serum2_preset_writer import asset_references, overlay_vector
from core.serum2_state_reconstruct import (
    DEFAULT_RENDER_STATE_DIR,
    decode_host_template,
    reconstruct_partial_vstpreset,
)
from core.serum2_targets import decode_vector
from core.train import load_parameter_model


OUTPUT_DIR = PROJECT_ROOT / "data" / "models" / "milestone4_export_gate"
REPORT = PROJECT_ROOT / "data" / "models" / "milestone4_export_report.json"


def _preset_path(preset_id: int) -> Path:
    with sqlite3.connect(DEFAULT_DB_PATH) as connection:
        row = connection.execute(
            "SELECT path FROM presets WHERE id=?", (preset_id,)
        ).fetchone()
    if row is None:
        raise KeyError(preset_id)
    return Path(row[0]).resolve()


def _render_loaded(engine: Any, processor: Any, midi_note: int = 60) -> np.ndarray:
    if hasattr(processor, "clear_midi"):
        processor.clear_midi()
    processor.add_midi_note(midi_note, 100, 0.0, 4.0)
    engine.render(4.0)
    audio = np.asarray(engine.get_audio(), dtype=np.float32)
    if audio.shape[0] != 2 and audio.shape[1] == 2:
        audio = audio.T
    mono = np.mean(audio, axis=0, dtype=np.float32)
    return librosa.resample(
        mono, orig_sr=44_100, target_sr=48_000, res_type="soxr_hq"
    ).astype(np.float32)


def _render_intended(
    verifier: PresetExportVerifier,
    candidate: Candidate,
    schema: dict[str, Any],
    temporary_dir: Path,
) -> np.ndarray:
    engine, processor = verifier.hosts[candidate.synth]
    if candidate.synth == "serum1":
        if processor.load_preset(str(_preset_path(candidate.base_preset_id))) is False:
            raise RuntimeError("Serum 1 rejected export-gate base preset")
        for index, value in enumerate(candidate.vector):
            processor.set_parameter(index, float(value))
    else:
        template = decode_host_template(
            (
                DEFAULT_RENDER_STATE_DIR
                / f"{candidate.base_preset_id}.vstpreset"
            ).read_bytes()
        )
        graph = decode_vector(candidate.vector, schema, candidate.mask)
        partial = Serum2Preset(
            path=temporary_dir / "candidate.SerumPreset",
            metadata={"presetName": "Export gate candidate"},
            data=graph,
            metadata_length=0,
            cbor_length=0,
            payload_version=0,
            compressed_length=0,
        )
        state, _partition = reconstruct_partial_vstpreset(
            partial, template, merge_matching_lists=True
        )
        path = temporary_dir / f"{candidate.base_preset_id}.vstpreset"
        path.write_bytes(state)
        if processor.load_vst3_preset(str(path)) is False:
            raise RuntimeError("Serum 2 rejected export-gate candidate state")
    return _render_loaded(engine, processor)


def _canonical_audio(preset_id: int) -> np.ndarray:
    audio, rate = sf.read(
        PROJECT_ROOT / "data" / "audio" / str(preset_id) / "60.wav",
        dtype="float32",
        always_2d=True,
    )
    mono = np.mean(audio, axis=1, dtype=np.float32)[: 4 * rate]
    return librosa.resample(
        mono, orig_sr=rate, target_sr=48_000, res_type="soxr_hq"
    ).astype(np.float32)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT_DIR.iterdir():
        if path.is_file():
            path.unlink()
    stores = {1: _serum1_targets(DEFAULT_DB_PATH), 2: _serum2_targets()}
    _model, checkpoint = load_parameter_model(device="cpu")
    schema = checkpoint["serum2_schema"]
    database = Database(DEFAULT_DB_PATH)
    verifier = PresetExportVerifier()
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(4404)
    try:
        with tempfile.TemporaryDirectory(prefix="patchlab-export-gate-") as temporary:
            temp = Path(temporary)
            for code, synth, extension in (
                (1, "serum1", ".fxp"),
                (2, "serum2", ".SerumPreset"),
            ):
                store = stores[code]
                ids = [
                    preset_id
                    for preset_id in store.preset_row
                    if (
                        PROJECT_ROOT
                        / "data"
                        / "audio"
                        / str(preset_id)
                        / "60.wav"
                    ).is_file()
                ][:5]
                if len(ids) != 5:
                    raise RuntimeError(f"Need five rendered {synth} presets")
                for index, preset_id in enumerate(ids):
                    source_row = store.preset_row[preset_id]
                    base = np.asarray(store.vectors[source_row], dtype=np.float32)
                    mask = np.asarray(store.masks[source_row], dtype=np.bool_)
                    modified = index >= 2
                    vector = base.copy()
                    if modified:
                        for _attempt in range(100):
                            if code == 1:
                                vector, _ = perturb_serum1(
                                    base, mask, store.mapping, rng
                                )
                                break
                            vector, _ = perturb_serum2(base, mask, schema, rng)
                            full = database.serum2_full_settings(preset_id)
                            _graph, applied, _skipped = overlay_vector(
                                full["settings"],
                                schema,
                                vector,
                                mask,
                                base,
                            )
                            if applied:
                                break
                        else:
                            raise RuntimeError(
                                f"Could not create an applicable Serum 2 perturbation for {preset_id}"
                            )
                    candidate = Candidate(
                        synth=synth,
                        base_preset_id=preset_id,
                        vector=vector,
                        mask=mask,
                        origin="cma" if modified else "retrieved-1",
                        exact_base=not modified,
                    )
                    target = (
                        _render_intended(verifier, candidate, schema, temp)
                        if modified
                        else _canonical_audio(preset_id)
                    )
                    path = OUTPUT_DIR / f"{synth}-{index + 1}{extension}"
                    verification = write_and_verify_native_preset(
                        path,
                        synth=synth,
                        base_preset_id=preset_id,
                        vector=vector,
                        mask=mask,
                        meaningfully_modified=modified,
                        midi_note=60,
                        target_audio=target,
                        expected_clap_similarity=1.0,
                        verifier=verifier,
                    )
                    base_assets = (
                        asset_references(
                            database.serum2_full_settings(preset_id)["settings"]
                        )
                        if synth == "serum2"
                        else ()
                    )
                    assets_retained = (
                        verification.export.asset_references == base_assets
                    )
                    item = {
                        "synth": synth,
                        "base_preset_id": preset_id,
                        "case": "optimized" if modified else "copied",
                        "mode": verification.export.mode,
                        "path": str(path),
                        "decoded_graph_equal": verification.decoded_graph_equal,
                        "max_parameter_delta": verification.max_parameter_delta,
                        "render_state_coverage": verification.render_state_coverage,
                        "clap_similarity": verification.clap_similarity,
                        "expected_clap_similarity": verification.expected_clap_similarity,
                        "similarity_delta": verification.similarity_delta,
                        "asset_reference_count": len(
                            verification.export.asset_references
                        ),
                        "assets_retained": assets_retained,
                        "applied_fields": verification.export.applied_fields,
                        "skipped_changed_fields": len(
                            verification.export.skipped_fields
                        ),
                        "pass": verification.passed and assets_retained,
                    }
                    rows.append(item)
                    print(
                        f"EXPORT_GATE={len(rows)}/10 {synth} {item['case']} "
                        f"CLAP={item['clap_similarity']:.6f} "
                        f"coverage={item['render_state_coverage']:.4f}",
                        flush=True,
                    )
    finally:
        verifier.close()
    report = {
        "count": len(rows),
        "by_synth": {
            synth: {
                "count": sum(row["synth"] == synth for row in rows),
                "copied": sum(
                    row["synth"] == synth and row["case"] == "copied"
                    for row in rows
                ),
                "optimized": sum(
                    row["synth"] == synth and row["case"] == "optimized"
                    for row in rows
                ),
                "mean_clap_similarity": float(
                    np.mean(
                        [
                            row["clap_similarity"]
                            for row in rows
                            if row["synth"] == synth
                        ]
                    )
                ),
            }
            for synth in ("serum1", "serum2")
        },
        "rows": rows,
        "gate_pass": len(rows) == 10 and all(row["pass"] for row in rows),
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("EXPORT_REPORT=" + json.dumps(report["by_synth"], sort_keys=True))
    print("EXPORT_GATE_PASS=" + str(report["gate_pass"]).lower())
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

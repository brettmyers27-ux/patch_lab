#!/usr/bin/env python3
"""Write and mandatorily verify the current Match a Sound recommendation."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.audio_input import decode_audio_file
from core.branding import generated_preset_name
from core.fxp import build_fxp, parse_fxp
from core.match_library import resolve_result_path
from core.preset_export import (
    commit_temporary_export,
    write_and_verify_native_preset,
)
from core.preset_scan import sha1_file
from core.serum2_preset import parse_serum2_preset
from core.serum2_preset_writer import (
    branded_serum2_metadata,
    encode_serum2_preset,
)


def export_factory_exact(result: dict, recommendation: dict, output: Path) -> dict:
    source_value = recommendation.get("factory_source_path")
    if not source_value:
        raise RuntimeError(
            "The matching factory preset is not installed locally, so it cannot be exported."
        )
    source = Path(str(source_value)).expanduser().resolve()
    expected_hash = str(recommendation["content_hash"])
    actual_hash = sha1_file(source)
    if actual_hash != expected_hash:
        raise RuntimeError(
            "The local factory preset changed after startup verification; export was stopped."
        )
    synth = str(recommendation["synth"])
    if synth == "serum1":
        parsed = parse_fxp(source)
        if not parsed.payload:
            raise RuntimeError("The local Serum 1 factory preset has no state chunk.")
        payload = build_fxp(
            parsed.payload,
            plugin_id=parsed.plugin_id,
            plugin_version=parsed.plugin_version,
            program_name=generated_preset_name("serum1"),
        )
    else:
        parsed = parse_serum2_preset(source)
        if not isinstance(parsed.data, dict) or not parsed.data:
            raise RuntimeError("The local Serum 2 factory preset has no decoded settings graph.")
        payload = encode_serum2_preset(
            branded_serum2_metadata(
                parsed.metadata,
                name=generated_preset_name("serum2"),
            ),
            parsed.data,
            parsed.payload_version,
        )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    if synth == "serum1":
        parse_fxp(output)
    elif not parse_serum2_preset(output).data:
        output.unlink(missing_ok=True)
        raise RuntimeError("Branded factory Serum 2 preset did not decode.")
    return {
        "path": str(output),
        "mode": "factory-exact-branded-copy",
        "clap_similarity": float(recommendation["clap_similarity"]),
        "expected_clap_similarity": float(recommendation["clap_similarity"]),
        "render_state_coverage": 1.0,
        "decoded_graph_equal": True,
        "asset_reference_count": 0,
        "content_hash_verified": True,
        "audio_rendered": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    recommendation = result.get("recommendation")
    if not isinstance(recommendation, dict):
        print("EXPORT_ERROR=There is no recommendation to export", flush=True)
        return 1
    final_output = args.output.expanduser().resolve()
    extension = ".fxp" if recommendation["synth"] == "serum1" else ".SerumPreset"
    try:
        with tempfile.TemporaryDirectory(
            prefix="patchlab-generated-preset-"
        ) as temporary_directory:
            temporary_path = Path(temporary_directory) / f"generated{extension}"
            if result.get("factory_only"):
                payload = export_factory_exact(
                    result,
                    recommendation,
                    temporary_path,
                )
                warning = None
            else:
                candidate = np.load(
                    resolve_result_path(
                        args.result, recommendation["candidate_path"]
                    )
                )
                source = result["source"]
                decoded = decode_audio_file(
                    resolve_result_path(args.result, source["path"]),
                    start_offset_s=float(source["start_offset_s"]),
                )
                target = decoded.mono[
                    : int(
                        round(
                            min(4.0, decoded.used_duration_s)
                            * decoded.sample_rate
                        )
                    )
                ]
                if decoded.sample_rate != 48_000:
                    target = librosa.resample(
                        target,
                        orig_sr=decoded.sample_rate,
                        target_sr=48_000,
                        res_type="soxr_hq",
                    ).astype(np.float32)
                verified = write_and_verify_native_preset(
                    temporary_path,
                    synth=str(recommendation["synth"]),
                    base_preset_id=int(recommendation["base_preset_id"]),
                    vector=np.asarray(candidate["vector"], dtype=np.float32),
                    mask=np.asarray(candidate["mask"], dtype=np.bool_),
                    meaningfully_modified=bool(
                        recommendation["meaningfully_modified"]
                    ),
                    midi_note=int(result["detected"]["midi_note"]),
                    target_audio=target,
                    expected_clap_similarity=float(
                        recommendation["clap_similarity"]
                    ),
                    name=generated_preset_name(str(recommendation["synth"])),
                    structural_overrides=(
                        json.loads(str(candidate["structural_overrides_json"].item()))
                        if "structural_overrides_json" in candidate.files
                        else dict(recommendation.get("structural_overrides") or {})
                    ),
                )
                warning = None
                if not verified.passed:
                    warning = (
                        "The saved preset reloaded correctly, but its verification "
                        f"render scored {verified.clap_similarity:.4f} CLAP instead "
                        f"of the preview's {verified.expected_clap_similarity:.4f}."
                    )
                payload = {
                    "path": str(temporary_path),
                    "mode": verified.export.mode,
                    "clap_similarity": verified.clap_similarity,
                    "expected_clap_similarity": (
                        verified.expected_clap_similarity
                    ),
                    "render_state_coverage": (
                        verified.render_state_coverage
                    ),
                    "decoded_graph_equal": verified.decoded_graph_equal,
                    "asset_reference_count": len(
                        verified.export.asset_references
                    ),
                }
            commit_temporary_export(temporary_path, final_output)
            temporary_removed = not temporary_path.exists()
    except Exception as exc:
        print(f"EXPORT_ERROR={type(exc).__name__}: {exc}", flush=True)
        return 1
    payload["path"] = str(final_output)
    payload["temporary_export_used"] = True
    payload["temporary_export_deleted"] = temporary_removed
    payload["verification_warning"] = warning
    print("EXPORT_RESULT=" + json.dumps(payload, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Write and mandatorily verify the current Match a Sound recommendation."""

from __future__ import annotations

import argparse
import json
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
from core.preset_export import commit_temporary_export
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


def existing_match_as_recommendation(result: dict, index: int) -> dict:
    """Describe one closest match so it exports through the same verified path.

    A closest match is an existing preset, so exporting it is the branded exact
    copy that ``export_factory_exact`` already performs for a factory-only
    recommendation -- not a second implementation.
    """

    rows = list(result.get("existing_matches") or [])
    if not 0 <= index < len(rows):
        raise RuntimeError("That closest match is no longer part of this result.")
    row = rows[index]
    source = row.get("source_path") or ""
    if not row.get("local_source_available") or not source:
        raise RuntimeError(
            f"{row.get('name') or 'This preset'} is not installed on this Mac, so "
            "PatchLab has no file to copy. Install the pack it came from, then retry."
        )
    return {
        "synth": str(row["synth"]),
        "content_hash": str(row["content_hash"]),
        "factory_source_path": str(source),
        "clap_similarity": float(row.get("similarity", 0.0)),
        "base_preset_id": int(row.get("preset_id") or 0),
    }


def _target_audio(result_path: Path, result: dict) -> np.ndarray:
    source = result["source"]
    decoded = decode_audio_file(
        resolve_result_path(result_path, source["path"]),
        start_offset_s=float(source["start_offset_s"]),
    )
    target = decoded.mono[: int(round(min(4.0, decoded.used_duration_s) * decoded.sample_rate))]
    if decoded.sample_rate != 48_000:
        target = librosa.resample(
            target, orig_sr=decoded.sample_rate, target_sr=48_000, res_type="soxr_hq"
        ).astype(np.float32)
    return target


class _Builder:
    """Build and fully verify one preset into a private path.

    Expensive inputs (the candidate, the decoded target, the plug-in host and
    CLAP model inside the verifier) are prepared once and reused by every
    attempt in this process.
    """

    def __init__(self, result_path: Path, result: dict, recommendation: dict) -> None:
        self.result_path = result_path
        self.result = result
        self.recommendation = recommendation
        self._verifier = None
        self._inputs: dict | None = None

    def close(self) -> None:
        if self._verifier is not None:
            self._verifier.close()

    def _prepared(self) -> dict:
        if self._inputs is None:
            candidate = np.load(
                resolve_result_path(self.result_path, self.recommendation["candidate_path"])
            )
            self._inputs = {
                "vector": np.asarray(candidate["vector"], dtype=np.float32),
                "mask": np.asarray(candidate["mask"], dtype=np.bool_),
                "structural_overrides": (
                    json.loads(str(candidate["structural_overrides_json"].item()))
                    if "structural_overrides_json" in candidate.files
                    else dict(self.recommendation.get("structural_overrides") or {})
                ),
                "target": _target_audio(self.result_path, self.result),
            }
        return self._inputs

    def __call__(self, tracker, staging: Path):
        from core.preset_save import PresetValidationError, StagedPreset, sha256_file, validate_preset_file

        recommendation = self.recommendation
        synth = str(recommendation["synth"])
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.unlink(missing_ok=True)
        if self.result.get("factory_only"):
            tracker.stage = "construct"
            payload = export_factory_exact(self.result, recommendation, staging)
            identity = {
                "synth": synth,
                "catalog_id": int(recommendation.get("base_preset_id") or 0),
                "content_hash": str(recommendation.get("content_hash", "")),
            }
            warning = None
            mode = payload["mode"]
        else:
            from core.preset_export import PresetExportVerifier, write_native_preset

            inputs = self._prepared()
            tracker.stage = "construct"
            export = write_native_preset(
                staging,
                synth=synth,
                base_preset_id=int(recommendation["base_preset_id"]),
                vector=inputs["vector"],
                mask=inputs["mask"],
                meaningfully_modified=bool(recommendation["meaningfully_modified"]),
                name=generated_preset_name(synth),
                structural_overrides=inputs["structural_overrides"],
            )
            tracker.stage = "validate"
            if self._verifier is None:
                self._verifier = PresetExportVerifier()
            verified = self._verifier.verify(
                export,
                intended_vector=inputs["vector"],
                midi_note=int(self.result["detected"]["midi_note"]),
                target_audio=inputs["target"],
                expected_clap_similarity=float(recommendation["clap_similarity"]),
            )
            if not verified.structurally_valid:
                raise PresetValidationError(
                    "the preset did not reload intact: "
                    f"decoded={verified.decoded_graph_equal}, "
                    f"coverage={verified.render_state_coverage:.3f}",
                    reason="reload_mismatch",
                )
            warning = None
            if not verified.passed:
                warning = (
                    "The saved preset reloaded correctly, but its verification "
                    f"render scored {verified.clap_similarity:.4f} CLAP instead "
                    f"of the preview's {verified.expected_clap_similarity:.4f}."
                )
            identity = export.base_identity.as_dict() if export.base_identity else {}
            mode = export.mode
            payload = {
                "mode": export.mode,
                "clap_similarity": verified.clap_similarity,
                "expected_clap_similarity": verified.expected_clap_similarity,
                "render_state_coverage": verified.render_state_coverage,
                "decoded_graph_equal": verified.decoded_graph_equal,
                "asset_reference_count": len(export.asset_references),
            }
        tracker.stage = "validate"
        checked = validate_preset_file(staging, synth=synth)
        payload = {key: value for key, value in payload.items() if key != "path"}
        payload["verification_warning"] = warning
        return StagedPreset(
            path=staging,
            sha256=sha256_file(staging),
            mode=mode,
            base_identity=identity,
            validation={**checked, "headless_reload": not self.result.get("factory_only")},
            payload=payload,
        )


def _save_incident(args, result: dict, recommendation: dict, final_output: Path) -> int:
    """The auto-save / Retry Saving Preset path: bounded, classified, recorded."""

    from core.preset_save import (
        SaveIncident,
        automatic_failure_message,
        manual_failure_message,
        run_save_lifecycle,
    )

    incident = SaveIncident.load(args.incident)
    if not incident.base_identity:
        incident.base_identity = {
            "synth": str(recommendation["synth"]),
            "catalog_id": int(recommendation.get("base_preset_id") or 0),
        }
    extension = ".fxp" if recommendation["synth"] == "serum1" else ".SerumPreset"
    builder = _Builder(args.result, result, recommendation)
    try:
        outcome = run_save_lifecycle(
            incident,
            build=builder,
            destination=final_output,
            extension=extension,
            trigger=args.trigger,
            max_attempts=args.attempts,
        )
    finally:
        builder.close()
    for attempt in incident.attempts:
        summary = {
            key: attempt.get(key)
            for key in ("attempt", "trigger", "stage", "ok", "failure_kind", "failure_reason",
                        "exception_type", "message", "elapsed_s")
        }
        print("EXPORT_ATTEMPT=" + json.dumps(summary, default=str), flush=True)
    if outcome.saved and outcome.final_path is not None:
        staged = outcome.incident.staged
        payload = {
            **dict(staged.get("payload") or {}),
            "path": str(outcome.final_path),
            "sha256": staged.get("sha256"),
            "incident_id": incident.incident_id,
            "attempts": outcome.attempts,
            "base_identity": incident.base_identity,
            "temporary_export_used": True,
        }
        # The file at its final name is now the only copy that matters.
        incident.discard_staged_artifact()
        incident.staged = {"sha256": payload["sha256"], "verified": True, "committed": True}
        incident.save()
        payload["temporary_export_deleted"] = True
        print("EXPORT_RESULT=" + json.dumps(payload, separators=(",", ":"), default=str), flush=True)
        return 0
    failure = outcome.failure
    message = (
        manual_failure_message(failure, unrecoverable=outcome.unrecoverable)
        if args.trigger == "manual"
        else automatic_failure_message(failure)
    )
    print("EXPORT_ERROR=" + message, flush=True)
    print(
        "EXPORT_FAILURE="
        + json.dumps(
            {
                "incident_id": incident.incident_id,
                "kind": failure.kind.value,
                "reason": failure.reason,
                "unrecoverable": outcome.unrecoverable,
                "attempts": outcome.attempts,
                "staged_available": incident.valid_staged_artifact() is not None,
                "user_message": message,
            }
        ),
        flush=True,
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--existing-match", type=int, default=None,
        help="Export this closest match (0-based) instead of the generated recommendation.",
    )
    parser.add_argument(
        "--incident", default=None,
        help="Save through this save incident (auto-save and Retry Saving Preset).",
    )
    parser.add_argument("--trigger", choices=("auto", "manual"), default="auto")
    parser.add_argument("--attempts", type=int, default=1)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    if args.existing_match is not None:
        try:
            recommendation = existing_match_as_recommendation(result, args.existing_match)
        except RuntimeError as exc:
            print(f"EXPORT_ERROR={exc}", flush=True)
            return 1
        result = {**result, "factory_only": True}
    else:
        recommendation = result.get("recommendation")
    if not isinstance(recommendation, dict):
        print("EXPORT_ERROR=There is no recommendation to export", flush=True)
        return 1
    final_output = args.output.expanduser().resolve()
    if args.incident:
        return _save_incident(args, result, recommendation, final_output)

    # A save the user aimed at a path they chose (Export Preset's save dialog,
    # closest-match export): one attempt, and a confirmed "Replace" replaces.
    from core.preset_save import StageTracker, validate_preset_file

    extension = ".fxp" if recommendation["synth"] == "serum1" else ".SerumPreset"
    builder = _Builder(args.result, result, recommendation)
    try:
        with tempfile.TemporaryDirectory(prefix="patchlab-generated-preset-") as temporary_directory:
            temporary_path = Path(temporary_directory) / f"generated{extension}"
            staged = builder(StageTracker(), temporary_path)
            commit_temporary_export(temporary_path, final_output)
            validate_preset_file(final_output, synth=str(recommendation["synth"]), expected_sha256=staged.sha256)
            temporary_removed = not temporary_path.exists()
    except Exception as exc:
        from core.worker_failure import report_worker_failure

        synth = str(recommendation.get("synth", ""))
        detail = report_worker_failure(
            exc, subsystem="preset-export", operation="saving the preset",
            synth=synth, requested_renderer=synth, destination=str(args.output),
        )
        print("EXPORT_ERROR=" + detail["user_message"], flush=True)
        print("EXPORT_ERROR_DETAIL=" + json.dumps(
            {k: v for k, v in detail.items() if k != "traceback"}, default=str), flush=True)
        return 1
    finally:
        builder.close()
    payload = dict(staged.payload)
    payload["path"] = str(final_output)
    payload["base_identity"] = staged.base_identity
    payload["temporary_export_used"] = True
    payload["temporary_export_deleted"] = temporary_removed
    print("EXPORT_RESULT=" + json.dumps(payload, separators=(",", ":"), default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

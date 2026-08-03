#!/usr/bin/env python3
"""Audio-verify each non-automatable Serum 2 structural category."""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import librosa
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from core.platform_env import ENV
from core.plugin_host import make_dawdreamer_processor, render_dawdreamer_note
from core.serum2_preset import Serum2Preset, parse_serum2_preset
from core.serum2_state_reconstruct import decode_host_template, reconstruct_partial_vstpreset
from core.synthesis_assets import resolve_synthesis_assets


DEFAULT_REPORT = PROJECT_ROOT / "data" / "stage3a" / "structural-mutations.json"


def _paths(node: Any, key_name: str, prefix: tuple[Any, ...] = ()) -> Iterable[tuple[tuple[Any, ...], Any]]:
    if isinstance(node, dict):
        for key, value in node.items():
            path = prefix + (key,)
            if key == key_name:
                yield path, value
            yield from _paths(value, key_name, path)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _paths(value, key_name, prefix + (index,))


def _get(graph: Any, path: tuple[Any, ...]) -> Any:
    cursor = graph
    for part in path:
        cursor = cursor[part]
    return cursor


def _set(graph: Any, path: tuple[Any, ...], value: Any) -> None:
    cursor = graph
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = value


def _features(audio: np.ndarray) -> dict[str, float | np.ndarray]:
    y = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(y)))
    normalized = y / max(peak, 1e-7)
    spectrum = np.abs(librosa.stft(normalized, n_fft=1024, hop_length=256))
    log_spectrum = np.log1p(spectrum)
    centroid = librosa.feature.spectral_centroid(S=spectrum, sr=44_100)[0]
    rms = librosa.feature.rms(y=normalized, frame_length=1024, hop_length=256)[0]
    tail_start = max(1, int(len(rms) * 0.7))
    return {
        "log_spectrum": log_spectrum,
        "centroid": centroid,
        "rms": rms,
        "centroid_std": float(np.std(centroid) / 22_050.0),
        "rms_std": float(np.std(rms)),
        "tail_ratio": float(np.mean(rms[tail_start:]) / max(np.mean(rms[:tail_start]), 1e-7)),
    }


def _distances(before: np.ndarray, after: np.ndarray, category: str) -> dict[str, Any]:
    left = _features(before)
    right = _features(after)
    spectral = float(np.mean(np.abs(left["log_spectrum"] - right["log_spectrum"])))
    waveform = float(np.sqrt(np.mean(np.square(before - after, dtype=np.float64))))
    centroid_shift = float(abs(np.mean(left["centroid"]) - np.mean(right["centroid"])) / 22_050.0)
    trajectory_shift = float(
        abs(float(left["centroid_std"]) - float(right["centroid_std"]))
        + abs(float(left["rms_std"]) - float(right["rms_std"]))
    )
    tail_shift = float(abs(np.log1p(left["tail_ratio"]) - np.log1p(right["tail_ratio"])))
    audible = spectral > 0.004 and waveform > 1e-4
    if category in {"wavetable", "embedded_wavetable"}:
        directional = audible and centroid_shift > 0.001
        criterion = "spectral-centroid shift accompanies the waveform change"
    elif category == "noise_sample":
        directional = audible and centroid_shift > 0.002
        criterion = "noise-band centroid shifts with sample identity"
    elif category == "mod_route":
        directional = audible and trajectory_shift > 0.001
        criterion = "amplitude/spectral trajectory modulation changes"
    else:
        directional = audible and (tail_shift > 0.005 or centroid_shift > 0.002)
        criterion = "effect tail or spectral fingerprint changes"
    return {
        "spectral_distance": spectral,
        "waveform_rms_difference": waveform,
        "centroid_shift_fraction_nyquist": centroid_shift,
        "trajectory_shift": trajectory_shift,
        "tail_log_ratio_shift": tail_shift,
        "audibly_changed": audible,
        "directionally_correct": directional,
        "directional_criterion": criterion,
    }


def _record(preset: Serum2Preset, path: tuple[Any, ...], replacement: Any, preset_id: int) -> dict[str, Any]:
    return {
        "preset": preset,
        "preset_id": preset_id,
        "path": path,
        "original": _get(preset.data, path),
        "replacement": replacement,
    }


def _candidate_records(package: Path, assets: Any) -> dict[str, list[dict[str, Any]]]:
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    rows = [row for row in manifest["presets_by_hash"].values() if row.get("synth") == "serum2"]
    parsed: list[tuple[int, Serum2Preset]] = []
    for row in rows:
        preset_id = int(row["preset_id"])
        if assets.find_render_state(preset_id) is not None:
            parsed.append((preset_id, parse_serum2_preset(package / row["relative_path"])))

    wt_values: list[Any] = []
    noise_values: list[Any] = []
    embedded_values: list[Any] = []
    sources: list[Any] = []
    for _preset_id, preset in parsed:
        wt_values.extend(value for _path, value in _paths(preset.data, "relativePathToWT"))
        noise_values.extend(value for _path, value in _paths(preset.data, "relativePathToNoiseSample"))
        embedded_values.extend(value for _path, value in _paths(preset.data, "embeddedWTData"))
        sources.extend(value for _path, value in _paths(preset.data, "source") if isinstance(value, list) and len(value) == 2)

    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for preset_id, preset in parsed:
        for path, value in _paths(preset.data, "relativePathToWT"):
            parent = _get(preset.data, path[:-2]) if len(path) >= 2 else {}
            if isinstance(parent, dict) and "embeddedWTData" in parent:
                continue
            replacement = next((item for item in wt_values if item != value), None)
            if replacement is not None:
                output["wavetable"].append(_record(preset, path, replacement, preset_id))
                break
        for path, value in _paths(preset.data, "embeddedWTData"):
            replacement = next(
                (item for item in embedded_values if type(item) is type(value) and item != value),
                None,
            )
            if replacement is not None:
                output["embedded_wavetable"].append(_record(preset, path, replacement, preset_id))
                break
        for path, value in _paths(preset.data, "relativePathToNoiseSample"):
            replacement = next((item for item in noise_values if item != value), None)
            if replacement is not None:
                output["noise_sample"].append(_record(preset, path, replacement, preset_id))
                break
        for key, slot in preset.data.items():
            if not key.startswith("ModSlot") or not isinstance(slot, dict) or "source" not in slot:
                continue
            amount = slot.get("plainParams", {}).get("kParamAmount", 0.0) if isinstance(slot.get("plainParams"), dict) else 0.0
            replacement = next((item for item in sources if item != slot["source"]), None)
            if replacement is not None and abs(float(amount)) >= 20.0:
                output["mod_route"].append(_record(preset, (key, "source"), replacement, preset_id))
                break
        for rack_name, rack in preset.data.items():
            if not rack_name.startswith("FXRack") or not isinstance(rack, dict):
                continue
            effects = rack.get("FX")
            if not isinstance(effects, list):
                continue
            for index, effect in enumerate(effects):
                if isinstance(effect, dict) and isinstance(effect.get("type"), int):
                    replacement = 6 if effect["type"] != 6 else 0
                    output["fx_type"].append(
                        _record(preset, (rack_name, "FX", index, "type"), replacement, preset_id)
                    )
                    break
            if output["fx_type"] and output["fx_type"][-1]["preset_id"] == preset_id:
                break
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=10)
    parser.add_argument("--attempt-limit", type=int, default=30)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--midi-note", type=int, default=60)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.per_category < 10:
        raise ValueError("Stage 3A requires at least 10 real presets per category")

    os.environ.setdefault("PATCHLAB_DISTRIBUTION_MODE", "1")
    assets = resolve_synthesis_assets()
    candidates = _candidate_records(args.package.expanduser().resolve(), assets)
    plugin = next(item for item in ENV.plugins_for("serum2") if item.format == "VST3" and item.hostable)
    engine, processor = make_dawdreamer_processor(plugin)
    results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with tempfile.TemporaryDirectory(prefix="patchlab-stage3a-mutations-") as scratch:
        state_path = Path(scratch) / "candidate.vstpreset"
        for category in ("wavetable", "embedded_wavetable", "noise_sample", "mod_route", "fx_type"):
            for record in candidates[category][: args.attempt_limit]:
                preset = record["preset"]
                template_path = assets.find_render_state(record["preset_id"])
                if template_path is None:
                    continue
                template = decode_host_template(template_path.read_bytes())
                pair: list[np.ndarray] = []
                loaded = True
                coverage = 0.0
                for replacement in (record["original"], record["replacement"]):
                    graph = copy.deepcopy(preset.data)
                    _set(graph, record["path"], copy.deepcopy(replacement))
                    candidate = Serum2Preset(preset.path, preset.metadata, graph, 0, 0, 2, 0)
                    blob, partition = reconstruct_partial_vstpreset(candidate, template, merge_matching_lists=True)
                    coverage = partition.coverage
                    state_path.write_bytes(blob)
                    if processor.load_vst3_preset(str(state_path)) is False:
                        loaded = False
                        break
                    stereo = render_dawdreamer_note(
                        engine,
                        processor,
                        midi_note=args.midi_note,
                        duration=args.duration,
                    )
                    pair.append(np.mean(stereo, axis=0).astype(np.float32, copy=False))
                metrics: dict[str, Any] = {
                    "preset_id": record["preset_id"],
                    "path": ".".join(str(part) for part in record["path"]),
                    "loaded": loaded,
                    "partition_coverage": coverage,
                }
                if loaded and len(pair) == 2:
                    metrics.update(_distances(pair[0], pair[1], category))
                else:
                    metrics.update({"audibly_changed": False, "directionally_correct": False})
                results[category].append(metrics)
                if sum(bool(item["audibly_changed"]) for item in results[category]) >= args.per_category:
                    break

    summary: dict[str, Any] = {}
    hard_gate_passed = True
    for category, records in results.items():
        changed = sum(bool(item["audibly_changed"]) for item in records)
        summary[category] = {
            "attempted": len(records),
            "loaded": sum(bool(item["loaded"]) for item in records),
            "audibly_changed": changed,
            "directionally_correct": sum(bool(item["directionally_correct"]) for item in records),
            "median_spectral_distance": float(np.median([item.get("spectral_distance", 0.0) for item in records])),
        }
        hard_gate_passed &= changed >= args.per_category
    payload = {
        "status": "pass" if hard_gate_passed else "fail",
        "hard_gate_passed": hard_gate_passed,
        "per_category_required": args.per_category,
        "duration_seconds": args.duration,
        "summary": summary,
        "trials": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("STAGE3A_STRUCTURAL_MUTATIONS=" + json.dumps({"status": payload["status"], "summary": summary}, sort_keys=True))
    return 0 if hard_gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

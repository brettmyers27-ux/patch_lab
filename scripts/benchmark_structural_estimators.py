#!/usr/bin/env python3
"""Held-out evaluation for Serum 2 structural shortlist estimators."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from core.dataset import _serum2_targets
from core.matcher import Candidate, _init_render_worker, _render_candidate
from core.serum2_preset import parse_serum2_preset
from core.structural_estimators import (
    NearestStructuralEstimator,
    audio_descriptor,
    deterministic_split,
    evaluate_estimator,
)
from core.synthesis_assets import resolve_synthesis_assets


DEFAULT_REPORT = PROJECT_ROOT / "data" / "stage3a" / "structural-estimators.json"


def _primary_labels(data: dict[str, Any]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for index in range(3):
        wt = data.get(f"Oscillator{index}", {}).get(f"WTOsc{index}", {})
        if "relativePathToWT" in wt and "wavetable" not in labels:
            labels["wavetable"] = str(wt["relativePathToWT"])
    noise = data.get("Oscillator3", {}).get("NoiseOsc3", {})
    if "relativePathToNoiseSample" in noise:
        labels["noise_sample"] = str(noise["relativePathToNoiseSample"])
    for rack_index in range(3):
        effects = data.get(f"FXRack{rack_index}", {}).get("FX", [])
        if effects and isinstance(effects[0], dict) and "type" in effects[0]:
            labels["fx_type"] = str(effects[0]["type"])
            break
    for slot_index in range(64):
        route = data.get(f"ModSlot{slot_index}", {})
        if isinstance(route, dict) and "source" in route and route.get("destModuleParamName"):
            labels["mod_route"] = json.dumps(
                {"source": route["source"], "destination": route["destModuleParamName"]},
                sort_keys=True,
                separators=(",", ":"),
            )
            break
    return labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=0.75)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    os.environ.setdefault("PATCHLAB_DISTRIBUTION_MODE", "1")
    assets = resolve_synthesis_assets()
    store = _serum2_targets(assets.serum2_targets, assets.serum2_schema)
    package = args.package.expanduser().resolve()
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    labels_by_id: dict[int, dict[str, str]] = {}
    for row in manifest["presets_by_hash"].values():
        if row.get("synth") == "serum2":
            preset_id = int(row["preset_id"])
            labels_by_id[preset_id] = _primary_labels(
                parse_serum2_preset(package / row["relative_path"]).data
            )
    preset_ids = sorted(set(labels_by_id).intersection(store.preset_row))
    candidates = []
    for preset_id in preset_ids:
        row = store.preset_row[preset_id]
        candidates.append(
            Candidate(
                "serum2",
                preset_id,
                np.asarray(store.vectors[row], dtype=np.float32),
                np.asarray(store.masks[row], dtype=np.bool_),
                "stage3a-estimator-ground-truth",
                exact_base=True,
                midi_note=60,
            )
        )
    context = mp.get_context("spawn")
    waveforms: dict[int, np.ndarray] = {}
    failures: dict[int, str] = {}
    with tempfile.TemporaryDirectory(prefix="patchlab-stage3a-estimators-") as scratch:
        with context.Pool(args.processes, initializer=_init_render_worker, initargs=(scratch, assets)) as pool:
            for candidate, result in zip(
                candidates,
                pool.imap(_render_candidate, [(item, 60, args.duration) for item in candidates], chunksize=4),
                strict=True,
            ):
                waveform, _coverage, error = result
                if error or waveform is None:
                    failures[candidate.base_preset_id] = error or "missing waveform"
                else:
                    waveforms[candidate.base_preset_id] = waveform

    report: dict[str, Any] = {
        "status": "complete",
        "rendered": len(waveforms),
        "render_failures": failures,
        "split": "preset_id modulo 5; zero is held out",
        "estimators": {},
    }
    adopted: dict[str, Any] = {}
    for category in ("fx_type", "wavetable", "mod_route", "noise_sample"):
        ids = [preset_id for preset_id in preset_ids if preset_id in waveforms and category in labels_by_id[preset_id]]
        modes = "noise" if category == "noise_sample" else "full"
        features = np.stack([audio_descriptor(waveforms[preset_id], mode=modes) for preset_id in ids])
        labels = [labels_by_id[preset_id][category] for preset_id in ids]
        train, test = deterministic_split(ids)
        estimator = NearestStructuralEstimator(mode=modes).fit(features[train], [labels[index] for index in train])
        metrics = evaluate_estimator(estimator, features[test], [labels[index] for index in test])
        result = {
            "samples": metrics.samples,
            "classes": metrics.classes,
            "top1": metrics.top1,
            "top5": metrics.top5,
            "common_top1": metrics.common_top1,
            "common_top5": metrics.common_top5,
            "adopted": metrics.adopted,
            "decision": "retain" if metrics.adopted else "drop; did not beat most-common top-1",
        }
        report["estimators"][category] = result
        if metrics.adopted:
            adopted[category] = {
                "features": estimator.features.tolist(),
                "labels": estimator.labels,
                "mode": estimator.mode,
            }
    report["adopted_estimators"] = sorted(adopted)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    model_path = args.report.parent / "structural-estimator-index.json"
    model_path.write_text(json.dumps(adopted, separators=(",", ":")) + "\n", encoding="utf-8")
    print("STAGE3A_STRUCTURAL_ESTIMATORS=" + json.dumps(report["estimators"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

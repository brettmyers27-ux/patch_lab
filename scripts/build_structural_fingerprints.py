#!/usr/bin/env python3
"""Build and gate Stage 3B controlled Serum 2 structural fingerprints.

All plug-in work is headless through DawDreamer.  The generated NPZ and detail
JSON live below ``data/`` and are private runtime artifacts.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import sqlite3
import tempfile
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import librosa
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from core.factory_bundle import DEFAULT_FACTORY_BUNDLE, FactoryBundle
from core.platform_env import ENV
from core.plugin_host import make_dawdreamer_processor, render_dawdreamer_note
from core.serum2_state_reconstruct import (
    HostStateTemplate,
    decode_host_template,
    encode_host_template,
)
from core.structural_estimators import (
    ControlledFingerprintIndex,
    controlled_descriptor,
    deterministic_split,
)


CATEGORIES = ("fx_type", "wavetable", "mod_route", "noise_sample")
BASE_PRESET_IDS = {
    "fx_type": 4423,
    "wavetable": 699,
    "mod_route": 699,
    "noise_sample": 699,
}
FIXED_BASELINES = {
    "fx_type": {"top1": 0.20714285714285716, "top5": 0.7285714285714285},
    "wavetable": {"top1": 0.35294117647058826, "top5": 0.5126050420168067},
    "mod_route": {"top1": 0.11428571428571428, "top5": 0.2571428571428571},
    "noise_sample": {"top1": 0.6, "top5": 0.6869565217391305},
}
DEFAULT_INDEX = PROJECT_ROOT / "data" / "models" / "serum2_structural_fingerprints.npz"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "stage3b" / "controlled-fingerprints.json"
_WORKER: dict[str, Any] = {}


@dataclass(frozen=True, slots=True)
class PrivateBenchmarkAssets:
    feature_dir: Path
    serum2_targets: Path
    serum2_schema: Path
    preset_index: Path
    note_index: Path
    render_states: Path
    library_db: Path
    render_state_roots: tuple[Path, ...]
    factory_mapping: Path | None = None

    def find_render_state(self, preset_id: int) -> Path | None:
        for root in self.render_state_roots:
            candidate = root / f"{preset_id}.vstpreset"
            if candidate.is_file():
                return candidate
        return None


def _private_assets() -> PrivateBenchmarkAssets:
    features = PROJECT_ROOT / "data" / "features"
    models = PROJECT_ROOT / "data" / "models"
    states = models / "serum2_render_states"
    return PrivateBenchmarkAssets(
        feature_dir=features,
        serum2_targets=features / "serum2_targets.npz",
        serum2_schema=models / "serum2_target_schema.json",
        preset_index=features / "preset_index.npy",
        note_index=features / "note_index.npy",
        render_states=states,
        library_db=models / "patchlab-synthesis-catalog.sqlite",
        render_state_roots=(states,),
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _label(category: str, value: Any) -> str:
    if category == "mod_route":
        return json.dumps(
            {
                "source": value["source"],
                "destination": value["destination"]["destModuleParamName"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    return str(value)


def _plain(graph: dict[str, Any], module: str, updates: Mapping[str, Any]) -> None:
    current = graph.get(module)
    if not isinstance(current, dict) or "plainParams" not in current:
        return
    values = current.get("plainParams")
    values = dict(values) if isinstance(values, dict) else {}
    values.update(updates)
    current["plainParams"] = values


def _neutralize(graph: dict[str, Any], category: str) -> None:
    """Make one dry, deterministic carrier while retaining valid topology."""

    for slot_index in range(64):
        _plain(graph, f"ModSlot{slot_index}", {"kParamAmount": 0.0})
    _plain(graph, "VoiceFilter0", {"kParamEnable": 0.0, "kParamFreq": 1.0})
    _plain(graph, "Oscillator0", {"kParamEnable": 1.0, "kParamVolume": 1.0})
    _plain(graph, "Oscillator1", {"kParamEnable": 1.0, "kParamVolume": 0.0})
    _plain(graph, "Oscillator2", {"kParamEnable": 1.0, "kParamVolume": 0.0})
    _plain(graph, "Oscillator3", {"kParamEnable": 1.0, "kParamVolume": 0.0})
    if category == "noise_sample":
        _plain(graph, "Oscillator0", {"kParamVolume": 0.0})
        _plain(graph, "Oscillator3", {"kParamVolume": 1.0})
    elif category == "mod_route":
        # Destination modules stay available; only the measured slot moves.
        _plain(graph, "VoiceFilter0", {"kParamEnable": 1.0, "kParamFreq": 0.5})
        _plain(graph, "ModSlot0", {"kParamAmount": 32.0})


def _apply_choice(
    graph: dict[str, Any], category: str, value: Any, fixed_effect: Mapping[str, Any]
) -> None:
    if category == "wavetable":
        oscillator = graph.get("Oscillator0")
        if isinstance(oscillator, dict) and isinstance(oscillator.get("WTOsc0"), dict):
            oscillator["WTOsc0"]["relativePathToWT"] = value
    elif category == "noise_sample":
        oscillator = graph.get("Oscillator3")
        if isinstance(oscillator, dict) and isinstance(oscillator.get("NoiseOsc3"), dict):
            oscillator["NoiseOsc3"]["relativePathToNoiseSample"] = value
    elif category == "fx_type":
        effects = graph.get("FXRack0", {}).get("FX", [])
        if effects and isinstance(effects[0], dict):
            effects[0]["type"] = int(value)
    elif category == "mod_route":
        slot = graph.get("ModSlot0")
        if not isinstance(slot, dict):
            return
        slot["source"] = copy.deepcopy(value["source"])
        for key, item in value["destination"].items():
            slot[key] = copy.deepcopy(item)
        _plain(graph, "ModSlot0", {"kParamAmount": 32.0})
    else:
        raise KeyError(category)


def _init_fingerprint_worker(
    state_paths: dict[str, str], fixed_effect: dict[str, Any], scratch: str
) -> None:
    plugin = next(
        item
        for item in ENV.plugins_for("serum2")
        if item.format == "VST3" and item.hostable
    )
    engine, processor = make_dawdreamer_processor(plugin)
    _WORKER.clear()
    _WORKER.update(
        {
            "engine": engine,
            "processor": processor,
            "templates": {
                category: decode_host_template(Path(path).read_bytes())
                for category, path in state_paths.items()
                if not category.startswith("mod_route::")
            },
            "state_paths": state_paths,
            "route_cache": OrderedDict(),
            "fixed_effect": fixed_effect,
            "state_path": Path(scratch) / f"fingerprint-{os.getpid()}.vstpreset",
        }
    )


def _worker_template(template_key: str) -> HostStateTemplate:
    template = _WORKER["templates"].get(template_key)
    if template is not None:
        return template
    cache: OrderedDict[str, HostStateTemplate] = _WORKER["route_cache"]
    if template_key in cache:
        cache.move_to_end(template_key)
        return cache[template_key]
    template = decode_host_template(Path(_WORKER["state_paths"][template_key]).read_bytes())
    cache[template_key] = template
    if len(cache) > 8:
        cache.popitem(last=False)
    return template


def _render_fingerprint(payload: tuple[str, str, Any, float, int, str]) -> tuple[str, str, np.ndarray | None, str | None, float]:
    category, identifier, value, duration, midi_note, template_key = payload
    started = time.perf_counter()
    try:
        source = _worker_template(template_key)
        template = copy.deepcopy(source)
        for graph in (template.component.data, template.controller.data):
            if isinstance(graph, dict):
                _neutralize(graph, category)
                _apply_choice(graph, category, value, _WORKER["fixed_effect"])
        state_path: Path = _WORKER["state_path"]
        state_path.write_bytes(encode_host_template(template))
        processor = _WORKER["processor"]
        if processor.load_vst3_preset(str(state_path)) is False:
            raise RuntimeError("Serum 2 rejected controlled state")
        stereo = render_dawdreamer_note(
            _WORKER["engine"], processor, midi_note=midi_note, duration=duration
        )
        mono = np.ascontiguousarray(np.mean(stereo, axis=0), dtype=np.float32)
        audio = librosa.resample(
            mono, orig_sr=44_100, target_sr=24_000, res_type="soxr_hq"
        ).astype(np.float32, copy=False)
        descriptor = controlled_descriptor(audio, category, 24_000)
        return category, identifier, descriptor, None, time.perf_counter() - started
    except Exception as exc:
        return category, identifier, None, f"{type(exc).__name__}: {exc}", time.perf_counter() - started


def _fixed_effect(state_path: Path) -> dict[str, Any]:
    template = decode_host_template(state_path.read_bytes())
    effects = template.component.data.get("FXRack0", {}).get("FX", [])
    if not effects or not isinstance(effects[0], dict):
        raise RuntimeError(f"Base state {state_path} has no fixed FX slot")
    return copy.deepcopy(effects[0])


def _factory_graphs(bundle_path: Path, library_db: Path) -> dict[int, dict[str, Any]]:
    bundle = FactoryBundle(bundle_path)
    by_hash = {preset.content_hash: preset for preset in bundle.presets() if preset.synth == "serum2"}
    with sqlite3.connect(library_db) as connection:
        rows = connection.execute(
            "SELECT id,content_hash FROM presets WHERE synth='serum2' ORDER BY id"
        ).fetchall()
    result: dict[int, dict[str, Any]] = {}
    for preset_id, content_hash in rows:
        preset = by_hash.get(str(content_hash))
        if preset is None:
            continue
        settings, _metadata, _payload_version = bundle.settings(preset.id)
        result[int(preset_id)] = settings
    return result


def _route_base_states(
    graphs: Mapping[int, dict[str, Any]], assets: Any
) -> tuple[dict[str, str], dict[tuple[str, int], str]]:
    """Pick one verified topology for every observed destination module."""

    candidates: dict[tuple[str, int], tuple[int, Path]] = {}
    for preset_id, graph in graphs.items():
        state_path = assets.find_render_state(preset_id)
        if state_path is None:
            continue
        for name, slot in graph.items():
            if not name.startswith("ModSlot") or not isinstance(slot, dict):
                continue
            module_type = slot.get("destModuleTypeString")
            module_id = slot.get("destModuleID")
            if module_type is None or not isinstance(module_id, int):
                continue
            destination = (str(module_type), int(module_id))
            size = state_path.stat().st_size
            current = candidates.get(destination)
            if current is None or size < current[0]:
                candidates[destination] = (size, state_path)
    state_paths: dict[str, str] = {}
    key_by_destination: dict[tuple[str, int], str] = {}
    for position, destination in enumerate(sorted(candidates)):
        key = f"mod_route::{position}"
        key_by_destination[destination] = key
        state_paths[key] = str(candidates[destination][1])
    return state_paths, key_by_destination


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


def _render_ground_truth(
    preset_ids: list[int], duration: float, processes: int, assets: Any
) -> tuple[dict[int, np.ndarray], dict[int, str], float]:
    waveforms: dict[int, np.ndarray] = {}
    failures: dict[int, str] = {}
    started = time.perf_counter()
    processor = _WORKER["processor"]
    engine = _WORKER["engine"]
    for position, preset_id in enumerate(preset_ids, start=1):
        try:
            state_path = assets.find_render_state(preset_id)
            if state_path is None:
                raise FileNotFoundError(f"No render state for {preset_id}")
            if processor.load_vst3_preset(str(state_path)) is False:
                raise RuntimeError("Serum 2 rejected exact held-out state")
            stereo = render_dawdreamer_note(engine, processor, midi_note=60, duration=duration)
            mono = np.ascontiguousarray(np.mean(stereo, axis=0), dtype=np.float32)
            waveforms[preset_id] = librosa.resample(
                mono, orig_sr=44_100, target_sr=24_000, res_type="soxr_hq"
            ).astype(np.float32, copy=False)
        except Exception as exc:
            failures[preset_id] = f"{type(exc).__name__}: {exc}"
        if position % 100 == 0 or position == len(preset_ids):
            print(f"STAGE3B_GROUND_TRUTH={position}/{len(preset_ids)}", flush=True)
    return waveforms, failures, time.perf_counter() - started


def _rank_labels(
    index: ControlledFingerprintIndex,
    category: str,
    descriptors: np.ndarray,
    top_k: int,
    log_priors: dict[str, float],
) -> list[list[str]]:
    label_by_id = dict(zip(index.stable_ids[category], index.labels[category], strict=True))
    return [
        [label_by_id[item.value] for item in index.rank_descriptor(category, row, top_k=top_k, log_priors=log_priors)]
        for row in descriptors
    ]


def _metrics(predictions: list[list[str]], truth: list[str]) -> tuple[float, float]:
    top1 = float(np.mean([bool(items) and items[0] == expected for items, expected in zip(predictions, truth, strict=True)]))
    top5 = float(np.mean([expected in items for items, expected in zip(predictions, truth, strict=True)]))
    return top1, top5


def _evaluate(
    index: ControlledFingerprintIndex,
    graphs: dict[int, dict[str, Any]],
    waveforms: dict[int, np.ndarray],
) -> tuple[dict[str, Any], dict[str, float], list[str]]:
    labels_by_id = {preset_id: _primary_labels(graph) for preset_id, graph in graphs.items()}
    report: dict[str, Any] = {}
    strengths: dict[str, float] = {}
    adopted: list[str] = []
    for category in CATEGORIES:
        ids = [
            preset_id
            for preset_id in sorted(labels_by_id)
            if preset_id in waveforms and category in labels_by_id[preset_id]
        ]
        descriptors = np.stack(
            [controlled_descriptor(waveforms[preset_id], category) for preset_id in ids]
        )
        labels = [labels_by_id[preset_id][category] for preset_id in ids]
        train, test = deterministic_split(ids)
        counts = Counter(labels[position] for position in train)
        maximum = max(counts.values())
        log_priors = {
            label: math.log(count / maximum) for label, count in counts.items()
        }
        best_strength = 0.0
        best_train = (-1.0, -1.0)
        for strength in (0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2):
            index.prior_strengths[category] = strength
            predictions = _rank_labels(index, category, descriptors[train], 5, log_priors)
            score = _metrics(predictions, [labels[position] for position in train])
            if score > best_train:
                best_train = score
                best_strength = strength
        strengths[category] = best_strength
        index.prior_strengths[category] = best_strength
        predictions = _rank_labels(index, category, descriptors[test], 5, log_priors)
        top1, top5 = _metrics(predictions, [labels[position] for position in test])
        baseline = FIXED_BASELINES[category]
        passed = top1 > baseline["top1"]
        if passed:
            adopted.append(category)
        report[category] = {
            "heldout_samples": len(test),
            "classes": len(set(labels)),
            "top1": top1,
            "top5": top5,
            "fixed_common_top1": baseline["top1"],
            "fixed_common_top5": baseline["top5"],
            "beats_top1_baseline": passed,
            "beats_top5_baseline": top5 > baseline["top5"],
            "prior_strength_selected_on_training": best_strength,
            "training_top1": best_train[0],
            "training_top5": best_train[1],
            "decision": "retain" if passed else "drop; did not beat fixed Stage 3A common-value top-1",
        }
    return report, strengths, adopted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocabulary", type=Path, default=PROJECT_ROOT / "data" / "models" / "serum2_structural_space.json")
    parser.add_argument("--factory-bundle", type=Path, default=DEFAULT_FACTORY_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--midi-note", type=int, default=60)
    args = parser.parse_args()
    os.environ.setdefault("PATCHLAB_DISTRIBUTION_MODE", "1")
    stage_started = time.perf_counter()
    print("STAGE3B_SETUP=resolve-assets", flush=True)
    assets = _private_assets()
    print("STAGE3B_SETUP=load-vocabulary-and-graphs", flush=True)
    vocabulary = json.loads(args.vocabulary.read_text(encoding="utf-8"))
    graphs = _factory_graphs(args.factory_bundle, assets.library_db)
    if len(graphs) != 710:
        raise RuntimeError(f"Expected the same 710 Serum 2 factory graphs, found {len(graphs)}")
    state_paths = {}
    for category, preset_id in BASE_PRESET_IDS.items():
        path = assets.find_render_state(preset_id)
        if path is None:
            raise FileNotFoundError(f"Missing neutral base state {preset_id} for {category}")
        state_paths[category] = str(path)
    route_state_paths, route_keys = _route_base_states(graphs, assets)
    state_paths.update(route_state_paths)
    print(
        f"STAGE3B_SETUP=initialize-renderer route_topologies={len(route_state_paths)}",
        flush=True,
    )
    fixed_effect = _fixed_effect(Path(state_paths["fx_type"]))
    features: dict[str, np.ndarray] = {}
    stable_ids: dict[str, list[str]] = {}
    labels: dict[str, list[str]] = {}
    phase1: dict[str, Any] = {}
    context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="patchlab-stage3b-fingerprints-") as scratch:
        pool = None
        try:
            if args.processes == 1:
                _init_fingerprint_worker(state_paths, fixed_effect, scratch)
            else:
                pool = context.Pool(
                    args.processes,
                    initializer=_init_fingerprint_worker,
                    initargs=(state_paths, fixed_effect, scratch),
                )
            for category in CATEGORIES:
                print(f"STAGE3B_CATEGORY_START={category}", flush=True)
                entries = list(vocabulary["categories"][category]["entries"])
                payloads = [
                    (
                        category,
                        str(entry["id"]),
                        entry["value"],
                        args.duration,
                        args.midi_note,
                        (
                            route_keys[
                                (
                                    str(entry["value"]["destination"]["destModuleTypeString"]),
                                    int(entry["value"]["destination"]["destModuleID"]),
                                )
                            ]
                            if category == "mod_route"
                            else category
                        ),
                    )
                    for entry in entries
                ]
                if category == "mod_route":
                    payloads.sort(key=lambda item: (item[-1], item[1]))
                started = time.perf_counter()
                completed: dict[str, np.ndarray] = {}
                failures: dict[str, str] = {}
                seconds: list[float] = []
                results = (
                    map(_render_fingerprint, payloads)
                    if pool is None
                    else pool.imap(_render_fingerprint, payloads, chunksize=4)
                )
                for position, result in enumerate(results, start=1):
                    _category, identifier, descriptor, error, elapsed = result
                    seconds.append(elapsed)
                    if error or descriptor is None:
                        failures[identifier] = error or "missing descriptor"
                    else:
                        completed[identifier] = descriptor
                    if position % 100 == 0 or position == len(entries):
                        print(f"STAGE3B_FINGERPRINT={category}:{position}/{len(entries)}", flush=True)
                ordered = [entry for entry in entries if str(entry["id"]) in completed]
                stable_ids[category] = [str(entry["id"]) for entry in ordered]
                labels[category] = [_label(category, entry["value"]) for entry in ordered]
                features[category] = np.stack([completed[str(entry["id"])] for entry in ordered])
                phase1[category] = {
                    "enumerated": len(entries),
                    "rendered": len(ordered),
                    "failed": len(failures),
                    "failures": failures,
                    "wall_clock_s": time.perf_counter() - started,
                    "median_render_s": float(np.median(seconds)) if seconds else None,
                    "base_preset_id": (
                        None if category == "mod_route" else BASE_PRESET_IDS[category]
                    ),
                    "route_topology_carriers": (
                        len(route_state_paths) if category == "mod_route" else None
                    ),
                }
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    neutral_patch = {
        "shared": {
            "midi_note": args.midi_note,
            "duration_s": args.duration,
            "filter": "disabled except mod-route carrier",
            "fx_racks": (
                "empty in the wavetable/noise carrier; fixed existing validated topology "
                "for FX and route carriers because Serum rejects topology-list rewrites"
            ),
            "modulation": "all depths zero except ModSlot0=32 for mod_route",
            "envelopes_and_source_generators": "fixed from the named base state",
        },
        "wavetable": "preset 699 dry Osc0 at unity; Osc1/2/noise silent; only Osc0 WT path varies",
        "noise_sample": "preset 699 dry noise at unity; Osc0/1/2 silent; only noise path varies",
        "fx_type": "preset 4423 dry carrier and one validated FXRack0 slot; only its integer type varies; every setting leaf remains identical",
        "mod_route": (
            "ModSlot0 fixed depth 32; source and four destination identity leaves vary. "
            f"Serum validates destination topology, so {len(route_state_paths)} deterministic "
            "neutralized base states cover mutually exclusive destination module type/ID pairs; "
            "one universal base state is not loadable."
        ),
    }
    index = ControlledFingerprintIndex(
        features,
        stable_ids,
        labels,
        metadata={
            "schema_version": 1,
            "vocabulary": args.vocabulary.name,
            "neutral_patch": neutral_patch,
        },
    )
    index.save(args.output)
    waveforms, ground_failures, ground_seconds = _render_ground_truth(
        sorted(graphs), args.duration, args.processes, assets
    )
    estimator_report, strengths, adopted = _evaluate(index, graphs, waveforms)
    index.adopted = frozenset(adopted)
    index.prior_strengths = strengths
    index.metadata.update({"estimator_gate": estimator_report})
    index.save(args.output)
    payload = {
        "status": "complete",
        "phase1": phase1,
        "neutral_patch": neutral_patch,
        "artifact": str(args.output.resolve()),
        "artifact_bytes": args.output.stat().st_size,
        "phase2": {
            "heldout_split": "same Stage 3A preset_id modulo 5; zero is held out",
            "ground_truth_rendered": len(waveforms),
            "ground_truth_failures": ground_failures,
            "ground_truth_wall_clock_s": ground_seconds,
            "estimators": estimator_report,
            "adopted_estimators": adopted,
        },
        "wall_clock_s": time.perf_counter() - stage_started,
    }
    _atomic_json(args.report, payload)
    print(
        "STAGE3B_CONTROLLED_FINGERPRINTS="
        + json.dumps(
            {
                "counts": {category: row["rendered"] for category, row in phase1.items()},
                "estimators": estimator_report,
                "adopted": adopted,
                "wall_clock_s": payload["wall_clock_s"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

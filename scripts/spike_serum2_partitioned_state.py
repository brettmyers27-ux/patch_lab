#!/usr/bin/env python3
"""Final Serum 2 fidelity spike: partition merged CBOR into VST3 Comp/Cont state."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import cbor2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.db import DEFAULT_DB_PATH, Database
from core.platform_env import ENV
from core.plugin_host import (
    audio_levels,
    changed_parameter_count,
    dump_dawdreamer_parameters,
    dump_pedalboard_parameters,
    make_dawdreamer_processor,
    render_dawdreamer_note,
    render_pedalboard_note,
)
from core.serum2_preset import parse_serum2_preset
from core.serum2_state_reconstruct import (
    decode_host_template,
    reconstruct_vstpreset,
    structural_paths,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "models" / "serum2_partitioned_state_report.json"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "data" / "models" / "serum2_partitioned_spike"
EXPECTED_ACTIVE_MOD_SLOTS = (13, 15, 3, 26, 21)


def candidate() -> Any:
    return next(item for item in ENV.plugins_for("serum2") if item.format == "VST3")


def leaf_values(value: Any, path: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        if not value:
            return {path: value}
        result: dict[str, Any] = {}
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            result.update(leaf_values(child, child_path))
        return result
    if isinstance(value, list):
        if not value:
            return {path: value}
        result = {}
        for index, child in enumerate(value):
            result.update(leaf_values(child, f"{path}.[{index}]"))
        return result
    return {path: value}


def state_comparison(before: Any, after: Any, requested: Any, prefix: str) -> dict[str, Any]:
    before_leaves = leaf_values(before)
    after_leaves = leaf_values(after)
    requested_leaves = leaf_values(requested)
    changed = {
        path
        for path in before_leaves.keys() | after_leaves.keys()
        if before_leaves.get(path) != after_leaves.get(path)
    }
    comparable = requested_leaves.keys() & after_leaves.keys()
    matched = {path for path in comparable if requested_leaves[path] == after_leaves[path]}
    requested_nondefault = {
        path
        for path, value in requested_leaves.items()
        if path not in before_leaves or before_leaves[path] != value
    }
    nondefault_matched = {
        path
        for path in requested_nondefault & after_leaves.keys()
        if requested_leaves[path] == after_leaves[path]
    }
    before_top = before if isinstance(before, dict) else {}
    after_top = after if isinstance(after, dict) else {}
    requested_top = requested if isinstance(requested, dict) else {}
    return {
        f"{prefix}_changed_leaf_count": len(changed),
        f"{prefix}_requested_leaf_count": len(requested_leaves),
        f"{prefix}_requested_leaf_matches": len(matched),
        f"{prefix}_requested_leaf_coverage": (
            len(matched) / len(requested_leaves) if requested_leaves else 0.0
        ),
        f"{prefix}_requested_nondefault_leaf_count": len(requested_nondefault),
        f"{prefix}_requested_nondefault_leaf_matches": len(nondefault_matched),
        f"{prefix}_requested_nondefault_leaf_coverage": (
            len(nondefault_matched) / len(requested_nondefault)
            if requested_nondefault
            else 1.0
        ),
        f"{prefix}_changed_top_level_keys": sorted(
            key
            for key in before_top.keys() | after_top.keys()
            if before_top.get(key) != after_top.get(key)
        ),
        f"{prefix}_requested_top_level_matches": sorted(
            key
            for key in requested_top.keys() & after_top.keys()
            if requested_top[key] == after_top[key]
        ),
    }


def selected_state(data: Any, predicate: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    return {key: value for key, value in data.items() if predicate(str(key))}


def vector_result(initial: list[Any], loaded: list[Any]) -> dict[str, Any]:
    vector = [float(item.norm_value) for item in loaded[:541]]
    changed_indices = [
        item.index
        for before, item in zip(initial[:541], loaded[:541])
        if abs(item.norm_value - before.norm_value) > 1e-4
    ]
    fx_changed = [index for index in changed_indices if 471 <= index <= 518]
    mod_changed = [index for index in changed_indices if 312 <= index <= 439]
    mod_slots = sorted({(index - 312) // 2 + 1 for index in mod_changed})
    return {
        "parameter_count": len(loaded),
        "changed_from_init": changed_parameter_count(initial, loaded),
        "changed_core_indices": changed_indices,
        "fx_changed_indices": fx_changed,
        "fx_changed_count": len(fx_changed),
        "mod_changed_indices": mod_changed,
        "mod_changed_count": len(mod_changed),
        "mod_slots_changed": mod_slots,
        "mod_slots_changed_count": len(mod_slots),
        "core_vector": vector,
        "core_vector_sha256": hashlib.sha256(
            np.asarray(vector, dtype="<f4").tobytes()
        ).hexdigest(),
    }


def daw_worker(path: Path) -> dict[str, Any]:
    engine, processor = make_dawdreamer_processor(candidate())
    initial = dump_dawdreamer_parameters(processor)
    returned = processor.load_vst3_preset(str(path.resolve()))
    loaded = dump_dawdreamer_parameters(processor)
    audio = render_dawdreamer_note(engine, processor)
    loaded_after_render = dump_dawdreamer_parameters(processor)
    peak, rms = audio_levels(audio)
    after_render = vector_result(initial, loaded_after_render)
    return {
        "host": "dawdreamer",
        "api": "load_vst3_preset(filepath)",
        "return_value": returned,
        **vector_result(initial, loaded),
        "changed_from_init_after_render": after_render["changed_from_init"],
        "core_vector_after_render_sha256": after_render["core_vector_sha256"],
        "peak_dbfs": peak,
        "rms_dbfs": rms,
        "non_silent": rms > -60.0,
    }


def pedalboard_worker(path: Path) -> dict[str, Any]:
    from pedalboard import load_plugin

    processor = load_plugin(str(candidate().path), plugin_name="Serum 2")
    preset_data_before = bytes(processor.preset_data)
    raw_state_before = bytes(processor.raw_state)
    initial = dump_pedalboard_parameters(processor)
    returned = processor.load_preset(str(path.resolve()))
    preset_data_after = bytes(processor.preset_data)
    raw_state_after = bytes(processor.raw_state)
    decoded_before = decode_host_template(preset_data_before)
    decoded_after = decode_host_template(preset_data_after)
    decoded_requested = decode_host_template(path.read_bytes())

    def data_hash(value: Any) -> str:
        return hashlib.sha256(cbor2.dumps(value)).hexdigest()

    loaded = dump_pedalboard_parameters(processor)
    audio = render_pedalboard_note(processor)
    loaded_after_render = dump_pedalboard_parameters(processor)
    peak, rms = audio_levels(audio)
    after_render = vector_result(initial, loaded_after_render)
    return {
        "host": "pedalboard",
        "api": "load_preset(filepath)",
        "return_value": repr(returned),
        "preset_data_before_sha256": hashlib.sha256(preset_data_before).hexdigest(),
        "preset_data_after_sha256": hashlib.sha256(preset_data_after).hexdigest(),
        "preset_data_changed": preset_data_after != preset_data_before,
        "raw_state_before_sha256": hashlib.sha256(raw_state_before).hexdigest(),
        "raw_state_after_sha256": hashlib.sha256(raw_state_after).hexdigest(),
        "raw_state_changed": raw_state_after != raw_state_before,
        "component_data_before_sha256": data_hash(decoded_before.component.data),
        "component_data_after_sha256": data_hash(decoded_after.component.data),
        "component_data_requested_sha256": data_hash(decoded_requested.component.data),
        "component_data_changed": decoded_after.component.data != decoded_before.component.data,
        "component_data_matches_requested": (
            decoded_after.component.data == decoded_requested.component.data
        ),
        "controller_data_before_sha256": data_hash(decoded_before.controller.data),
        "controller_data_after_sha256": data_hash(decoded_after.controller.data),
        "controller_data_requested_sha256": data_hash(decoded_requested.controller.data),
        "controller_data_changed": decoded_after.controller.data != decoded_before.controller.data,
        "controller_data_matches_requested": (
            decoded_after.controller.data == decoded_requested.controller.data
        ),
        "preset_name_before": decoded_before.controller.metadata.get("presetName"),
        "preset_name_after": decoded_after.controller.metadata.get("presetName"),
        "preset_name_requested": decoded_requested.controller.metadata.get("presetName"),
        **state_comparison(
            decoded_before.component.data,
            decoded_after.component.data,
            decoded_requested.component.data,
            "component",
        ),
        **state_comparison(
            decoded_before.controller.data,
            decoded_after.controller.data,
            decoded_requested.controller.data,
            "controller",
        ),
        **state_comparison(
            selected_state(decoded_before.component.data, lambda key: key.startswith("ModSlot")),
            selected_state(decoded_after.component.data, lambda key: key.startswith("ModSlot")),
            selected_state(decoded_requested.component.data, lambda key: key.startswith("ModSlot")),
            "mod_state",
        ),
        **state_comparison(
            selected_state(decoded_before.component.data, lambda key: key.startswith("FXRack")),
            selected_state(decoded_after.component.data, lambda key: key.startswith("FXRack")),
            selected_state(decoded_requested.component.data, lambda key: key.startswith("FXRack")),
            "fx_state",
        ),
        **vector_result(initial, loaded),
        "changed_from_init_after_render": after_render["changed_from_init"],
        "core_vector_after_render_sha256": after_render["core_vector_sha256"],
        "peak_dbfs": peak,
        "rms_dbfs": rms,
        "non_silent": rms > -60.0,
    }


def worker(host: str, path: Path) -> int:
    try:
        result = daw_worker(path) if host == "dawdreamer" else pedalboard_worker(path)
        print("WORKER_RESULT=" + json.dumps(result), flush=True)
        return 0
    except Exception as exc:
        print(
            "WORKER_RESULT=" + json.dumps({"type": type(exc).__name__, "exception": repr(exc)}),
            flush=True,
        )
        return 2


def run_isolated(host: str, path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            host,
            "--container",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    result: dict[str, Any] = {
        "exit_code": completed.returncode,
        "crashed_by_signal": -completed.returncode if completed.returncode < 0 else None,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    for line in completed.stdout.splitlines():
        if line.startswith("WORKER_RESULT="):
            result["result"] = json.loads(line.split("=", 1)[1])
    return result


def source_fx_count(data: Any) -> int:
    total = 0
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "FX" and isinstance(value, list):
                total += sum(isinstance(item, dict) and bool(item) for item in value)
            total += source_fx_count(value)
    elif isinstance(data, list):
        total += sum(source_fx_count(value) for value in data)
    return total


def mapped_vector(database: Database, source_path: Path) -> list[float]:
    with database.connect() as connection:
        row = connection.execute(
            "SELECT id FROM presets WHERE path=? AND synth='serum2'", (str(source_path),)
        ).fetchone()
        if row is None:
            raise KeyError(f"No catalog row for {source_path}")
        values = connection.execute(
            "SELECT norm_value FROM params WHERE preset_id=? ORDER BY param_index", (int(row["id"]),)
        ).fetchall()
    return [float(value[0]) for value in values[:541]]


def diff_indices(left: list[float], right: list[float], start: int = 0, end: int = 541) -> list[int]:
    return [
        index
        for index in range(start, min(end, len(left), len(right)))
        if abs(left[index] - right[index]) > 1e-4
    ]


def enrich_host_result(test: dict[str, Any], baseline: list[float]) -> None:
    result = test.get("result")
    if not isinstance(result, dict) or "core_vector" not in result:
        return
    vector = result["core_vector"]
    differences = diff_indices(vector, baseline)
    result["differs_from_mapped_indices"] = differences
    result["differs_from_mapped_count"] = len(differences)
    result["fx_differs_from_mapped"] = [index for index in differences if 471 <= index <= 518]
    result["mod_differs_from_mapped"] = [index for index in differences if 312 <= index <= 439]


def host_gate(tests: list[dict[str, Any]]) -> dict[str, Any]:
    results = [test.get("result") for test in tests]
    loaded = [result for result in results if isinstance(result, dict) and "core_vector" in result]
    hashes = {result["core_vector_sha256"] for result in loaded}
    return {
        "accepted": len(loaded),
        "all_changed_from_init": len(loaded) == len(tests)
        and all(result["changed_from_init"] >= 5 for result in loaded),
        "all_non_silent": len(loaded) == len(tests) and all(result["non_silent"] for result in loaded),
        "vectors_all_distinct": len(hashes) == len(tests),
        "section_6_1_pass": len(loaded) == len(tests)
        and all(result["changed_from_init"] >= 5 and result["non_silent"] for result in loaded)
        and len(hashes) == len(tests),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("presets", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--worker", choices=("dawdreamer", "pedalboard"))
    parser.add_argument("--container", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.container is None:
            parser.error("--container is required for worker mode")
        return worker(args.worker, args.container)
    if len(args.presets) != 5:
        parser.error("provide the five accepted Serum 2 fixture paths")

    from pedalboard import load_plugin

    live = load_plugin(str(candidate().path), plugin_name="Serum 2")
    template = decode_host_template(bytes(live.preset_data))
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    database = Database(args.db)
    reports = []
    for index, source_path in enumerate(args.presets, start=1):
        preset = parse_serum2_preset(source_path)
        container, partition = reconstruct_vstpreset(preset, template)
        container_path = args.artifact_dir / f"{index}.vstpreset"
        container_path.write_bytes(container)
        baseline = mapped_vector(database, source_path)
        daw = run_isolated("dawdreamer", container_path)
        pedalboard = run_isolated("pedalboard", container_path)
        enrich_host_result(daw, baseline)
        enrich_host_result(pedalboard, baseline)
        reports.append(
            {
                "source": str(source_path),
                "preset_name": preset.metadata.get("presetName", source_path.stem),
                "container": str(container_path),
                "container_size": len(container),
                "source_fx_modules": source_fx_count(preset.data),
                "expected_active_mod_slots": EXPECTED_ACTIVE_MOD_SLOTS[index - 1],
                "partition": {
                    "total_leaves": partition.total_leaves,
                    "matched_leaves": partition.matched_leaves,
                    "coverage": partition.coverage,
                    "component_leaf_count": len(partition.matched_component_paths),
                    "controller_leaf_count": len(partition.matched_controller_paths),
                    "both_leaf_count": len(partition.matched_both_paths),
                    "unmatched_paths": list(partition.unmatched_paths),
                },
                "dawdreamer": daw,
                "pedalboard": pedalboard,
            }
        )
        print(f"PARTITIONED_STATE_PROGRESS={index}/5", flush=True)

    payload = {
        "template": {
            "class_id": template.class_id,
            "component_metadata": template.component.metadata,
            "controller_metadata": template.controller.metadata,
            "component_top_keys": sorted(template.component.data),
            "controller_top_keys": sorted(template.controller.data),
            "component_structure_paths": sorted(structural_paths(template.component.data)),
            "controller_structure_paths": sorted(structural_paths(template.controller.data)),
        },
        "baseline": {"sample_mapping_coverage": 0.3731958762886598, "catalog_weighted_coverage": 0.40324104899160207},
        "presets": reports,
        "dawdreamer_gate": host_gate([report["dawdreamer"] for report in reports]),
        "pedalboard_gate": host_gate([report["pedalboard"] for report in reports]),
    }
    unmatched = Counter(
        path for report in reports for path in report["partition"]["unmatched_paths"]
    )
    payload["aggregate"] = {
        "mean_structural_coverage": sum(report["partition"]["coverage"] for report in reports) / 5,
        "unmatched_unique": len(unmatched),
        "unmatched_occurrences": sum(unmatched.values()),
        "unmatched_paths": dict(unmatched),
    }
    write_json(args.output, payload)
    print(json.dumps({key: payload[key] for key in ("aggregate", "dawdreamer_gate", "pedalboard_gate")}, indent=2))
    print(f"OUTPUT={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

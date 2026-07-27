#!/usr/bin/env python3
"""Isolated diagnostic for native Serum 2 state/preset loading APIs."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "models" / "serum2_native_state_diagnostic.json"


def candidate() -> Any:
    return next(item for item in ENV.plugins_for("serum2") if item.format == "VST3")


def blob_info(label: str, data: bytes) -> dict[str, Any]:
    result: dict[str, Any] = {
        "label": label,
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "first_64_hex": data[:64].hex(" "),
        "first_64_ascii": "".join(chr(value) if 32 <= value < 127 else "." for value in data[:64]),
        "starts_xferjson": data.startswith(b"XferJson\0"),
        "starts_vst3": data.startswith(b"VST3"),
    }
    if data.startswith(b"VC2!") and len(data) >= 8:
        result["container"] = "JUCE VC2 VST3PluginState"
        result["declared_xml_bytes"] = struct.unpack_from("<I", data, 4)[0]
        result["contains_component_tag"] = b"<IComponent>" in data
        result["contains_controller_tag"] = b"<IEditController>" in data
    if data.startswith(b"XferJson\0") and len(data) >= 17:
        metadata_size = struct.unpack_from("<Q", data, 9)[0]
        metadata_end = 17 + metadata_size
        result["metadata_size"] = metadata_size
        try:
            result["metadata"] = json.loads(data[17:metadata_end].decode("utf-8"))
        except Exception as exc:
            result["metadata_error"] = repr(exc)
        if metadata_end + 8 <= len(data):
            result["payload_length"] = struct.unpack_from("<I", data, metadata_end)[0]
            result["version_marker"] = struct.unpack_from("<I", data, metadata_end + 4)[0]
            result["zstd_magic_after_lengths"] = data[
                metadata_end + 8 : metadata_end + 12
            ] == b"\x28\xb5\x2f\xfd"
    if data.startswith(b"VST3") and len(data) >= 56:
        chunk_list_offset = struct.unpack_from("<Q", data, 40)[0]
        if data[chunk_list_offset : chunk_list_offset + 4] == b"List":
            count = struct.unpack_from("<I", data, chunk_list_offset + 4)[0]
            chunks = []
            for index in range(count):
                base = chunk_list_offset + 8 + index * 20
                chunk_id = data[base : base + 4].decode("ascii", "replace")
                offset, size = struct.unpack_from("<QQ", data, base + 4)
                chunk = data[offset : offset + size]
                child = blob_info(f"{label}:{chunk_id}", chunk)
                # Avoid redundant hashes/byte previews in the parent summary.
                chunks.append(
                    {
                        "id": chunk_id,
                        "offset": offset,
                        "size": size,
                        "starts_xferjson": child.get("starts_xferjson", False),
                        "metadata": child.get("metadata"),
                        "payload_length": child.get("payload_length"),
                        "version_marker": child.get("version_marker"),
                        "zstd_magic_after_lengths": child.get("zstd_magic_after_lengths"),
                    }
                )
            result["vst3_chunks"] = chunks
    return result


def vector_summary(initial: list[Any], loaded: list[Any]) -> dict[str, Any]:
    values = np.asarray([item.norm_value for item in loaded], dtype="<f4")
    changed = [
        item.index
        for item, before in zip(loaded, initial)
        if abs(item.norm_value - before.norm_value) > 1e-4
    ]
    return {
        "changed_from_init": changed_parameter_count(initial, loaded),
        "changed_indices_first_32": changed[:32],
        "vector_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
    }


def daw_load_path(path: Path) -> dict[str, Any]:
    engine, processor = make_dawdreamer_processor(candidate())
    initial = dump_dawdreamer_parameters(processor)
    result = processor.load_state(str(path.resolve()))
    loaded = dump_dawdreamer_parameters(processor)
    audio = render_dawdreamer_note(engine, processor)
    peak, rms = audio_levels(audio)
    return {
        "host": "dawdreamer",
        "api": "load_state(filepath: str) -> None",
        "return_value": repr(result),
        **vector_summary(initial, loaded),
        "peak_dbfs": peak,
        "rms_dbfs": rms,
        "non_silent": rms > -60.0,
    }


def pedalboard_load_path(path: Path) -> dict[str, Any]:
    from pedalboard import load_plugin

    processor = load_plugin(str(candidate().path), plugin_name="Serum 2")
    initial = dump_pedalboard_parameters(processor)
    result = processor.load_preset(str(path.resolve()))
    loaded = dump_pedalboard_parameters(processor)
    audio = render_pedalboard_note(processor)
    peak, rms = audio_levels(audio)
    return {
        "host": "pedalboard",
        "api": "load_preset(preset_file_path: str) -> None",
        "return_value": repr(result),
        **vector_summary(initial, loaded),
        "peak_dbfs": peak,
        "rms_dbfs": rms,
        "non_silent": rms > -60.0,
    }


def dump_live_state() -> dict[str, Any]:
    engine, processor = make_dawdreamer_processor(candidate())
    # DawDreamer requires the processor to have been activated/rendered before
    # save_state; otherwise it raises "Please load the plugin first!".
    render_dawdreamer_note(engine, processor, duration=0.2)
    with tempfile.NamedTemporaryFile(suffix=".state") as handle:
        processor.save_state(handle.name)
        handle.seek(0)
        daw_data = handle.read()
    from pedalboard import load_plugin

    pedalboard = load_plugin(str(candidate().path), plugin_name="Serum 2")
    raw_state = bytes(pedalboard.raw_state)
    preset_data = bytes(pedalboard.preset_data)
    return {
        "dawdreamer_save_state": blob_info("dawdreamer_save_state", daw_data),
        "pedalboard_raw_state": blob_info("pedalboard_raw_state", raw_state),
        "pedalboard_preset_data": blob_info("pedalboard_preset_data", preset_data),
    }


def pedalboard_assign(path: Path, property_name: str) -> dict[str, Any]:
    from pedalboard import load_plugin

    processor = load_plugin(str(candidate().path), plugin_name="Serum 2")
    initial = dump_pedalboard_parameters(processor)
    setattr(processor, property_name, path.read_bytes())
    loaded = dump_pedalboard_parameters(processor)
    audio = render_pedalboard_note(processor)
    peak, rms = audio_levels(audio)
    return {
        "host": "pedalboard",
        "api": f"{property_name} = raw_file_bytes",
        **vector_summary(initial, loaded),
        "peak_dbfs": peak,
        "rms_dbfs": rms,
        "non_silent": rms > -60.0,
    }


def callable_info(obj: Any) -> dict[str, Any]:
    try:
        signature = str(inspect.signature(obj))
    except Exception as exc:
        signature = f"unavailable: {type(exc).__name__}: {exc}"
    return {"signature": signature, "docstring": inspect.getdoc(obj)}


def api_surface() -> dict[str, Any]:
    _engine, processor = make_dawdreamer_processor(candidate())
    from pedalboard import load_plugin

    pedalboard = load_plugin(str(candidate().path), plugin_name="Serum 2")
    return {
        "dawdreamer": {
            "load_state": callable_info(processor.load_state),
            "save_state": callable_info(processor.save_state),
            "load_preset": callable_info(processor.load_preset),
            "load_vst3_preset": callable_info(processor.load_vst3_preset),
        },
        "pedalboard": {
            "load_preset": callable_info(pedalboard.load_preset),
            "raw_state": {
                "type": type(pedalboard.raw_state).__name__,
                "docstring": inspect.getdoc(type(pedalboard).__dict__["raw_state"]),
            },
            "preset_data": {
                "type": type(pedalboard.preset_data).__name__,
                "docstring": inspect.getdoc(type(pedalboard).__dict__["preset_data"]),
            },
        },
    }


def worker(mode: str, path: Path | None, property_name: str | None) -> int:
    operations = {
        "daw-load-path": lambda: daw_load_path(path),
        "pb-load-path": lambda: pedalboard_load_path(path),
        "dump-state": dump_live_state,
        "pb-assign": lambda: pedalboard_assign(path, str(property_name)),
    }
    try:
        result = operations[mode]()
        print("WORKER_RESULT=" + json.dumps(result, default=str), flush=True)
        return 0
    except Exception as exc:
        print(
            "WORKER_RESULT="
            + json.dumps({"exception": repr(exc), "type": type(exc).__name__}),
            flush=True,
        )
        return 2


def run_isolated(mode: str, path: Path | None = None, property_name: str | None = None) -> dict[str, Any]:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", mode]
    if path is not None:
        command += ["--preset", str(path)]
    if property_name is not None:
        command += ["--property", property_name]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=45)
    result: dict[str, Any] = {
        "mode": mode,
        "exit_code": completed.returncode,
        "crashed_by_signal": -completed.returncode if completed.returncode < 0 else None,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    for line in completed.stdout.splitlines():
        if line.startswith("WORKER_RESULT="):
            result["result"] = json.loads(line.split("=", 1)[1])
    return result


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("presets", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker", choices=("daw-load-path", "pb-load-path", "dump-state", "pb-assign"))
    parser.add_argument("--preset", type=Path)
    parser.add_argument("--property", choices=("raw_state", "preset_data"))
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker, args.preset, args.property)
    if not args.presets:
        parser.error("provide at least one untouched .SerumPreset file")

    payload: dict[str, Any] = {
        "api_surface": api_surface(),
        "source_presets": [blob_info(str(path), path.read_bytes()) for path in args.presets],
        "direct_path_tests": {"dawdreamer": [], "pedalboard": []},
    }
    for path in args.presets:
        payload["direct_path_tests"]["dawdreamer"].append(run_isolated("daw-load-path", path))
        payload["direct_path_tests"]["pedalboard"].append(run_isolated("pb-load-path", path))
    payload["live_state_dump"] = run_isolated("dump-state")

    dump_result = payload["live_state_dump"].get("result", {})
    retry: dict[str, Any] = {}
    if dump_result.get("pedalboard_raw_state", {}).get("starts_xferjson"):
        retry["pedalboard_raw_state"] = [
            run_isolated("pb-assign", path, "raw_state") for path in args.presets
        ]
    if dump_result.get("pedalboard_preset_data", {}).get("starts_xferjson"):
        retry["pedalboard_preset_data"] = [
            run_isolated("pb-assign", path, "preset_data") for path in args.presets
        ]
    payload["conditional_raw_byte_retries"] = retry
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, default=str))
    print(f"OUTPUT={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

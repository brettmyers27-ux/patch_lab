#!/usr/bin/env python3
"""Print a structural inventory for real decoded Serum 2 presets."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.serum2_preset import Serum2Preset, parse_serum2_preset


@dataclass(slots=True)
class Inventory:
    types: collections.Counter[str] = field(default_factory=collections.Counter)
    key_names: collections.Counter[str] = field(default_factory=collections.Counter)
    binary_fields: list[dict[str, Any]] = field(default_factory=list)
    largest_strings: list[tuple[int, str, str]] = field(default_factory=list)
    largest_lists: list[tuple[int, str]] = field(default_factory=list)
    largest_dicts: list[tuple[int, str]] = field(default_factory=list)
    parameter_fields: list[dict[str, Any]] = field(default_factory=list)
    candidate_fields: list[dict[str, Any]] = field(default_factory=list)


def path_text(parts: tuple[str, ...]) -> str:
    return ".".join(parts) if parts else "$"


def preview(value: Any, limit: int = 100) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def inventory_value(value: Any, result: Inventory, parts: tuple[str, ...] = ()) -> None:
    type_name = type(value).__name__
    result.types[type_name] += 1
    location = path_text(parts)
    if isinstance(value, dict):
        result.largest_dicts.append((len(value), location))
        for key, child in value.items():
            key_text = str(key)
            result.key_names[key_text] += 1
            lowered = key_text.casefold()
            item = {
                "path": path_text(parts + (key_text,)),
                "type": type(child).__name__,
                "size": len(child) if hasattr(child, "__len__") else None,
                "preview": preview(child),
            }
            if "param" in lowered:
                result.parameter_fields.append(item)
            if any(token in lowered for token in ("state", "blob", "binary", "wave", "table")):
                result.candidate_fields.append(item)
            inventory_value(child, result, parts + (key_text,))
    elif isinstance(value, (list, tuple)):
        result.largest_lists.append((len(value), location))
        for index, child in enumerate(value):
            inventory_value(child, result, parts + (f"[{index}]",))
    elif isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        result.binary_fields.append(
            {
                "path": location,
                "length": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "prefix_hex": raw[:32].hex(),
            }
        )
    elif isinstance(value, str):
        result.largest_strings.append((len(value), location, preview(value)))


def compact(items: Iterable[Any], count: int = 15) -> list[Any]:
    return list(items)[:count]


def walk(value: Any, parts: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    yield parts, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from walk(child, parts + (str(key),))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from walk(child, parts + (f"[{index}]",))


def compact_summary(preset: Serum2Preset) -> dict[str, Any]:
    type_counts: collections.Counter[str] = collections.Counter()
    plain_param_dicts = 0
    plain_param_values = 0
    plain_param_defaults = 0
    binary_fields: list[dict[str, Any]] = []
    embedded_fields: list[dict[str, Any]] = []
    path_references: list[dict[str, Any]] = []
    note_lists: list[dict[str, Any]] = []
    active_mod_slots = 0

    for parts, value in walk(preset.data):
        type_counts[type(value).__name__] += 1
        key = parts[-1] if parts else ""
        key_lower = key.casefold()
        location = path_text(parts)
        if key == "plainParams":
            if isinstance(value, dict):
                plain_param_dicts += 1
                plain_param_values += len(value)
            elif value == "default":
                plain_param_defaults += 1
        if isinstance(value, (bytes, bytearray, memoryview)):
            binary_fields.append({"path": location, "length": len(value)})
        if "embedded" in key_lower:
            embedded_fields.append(
                {
                    "path": location,
                    "type": type(value).__name__,
                    "length": len(value) if hasattr(value, "__len__") else None,
                }
            )
        if key.startswith("relativePath"):
            path_references.append({"path": location, "value": value})
        if key == "notes" and isinstance(value, list) and value:
            note_lists.append({"path": location, "length": len(value)})

    if isinstance(preset.data, dict):
        active_mod_slots = sum(
            1
            for key, value in preset.data.items()
            if key.startswith("ModSlot")
            and isinstance(value, dict)
            and value.get("plainParams") != "default"
        )
    metadata = preset.metadata if isinstance(preset.metadata, dict) else {}
    data_keys = list(preset.data) if isinstance(preset.data, dict) else []
    return {
        "file": preset.path.name,
        "preset_name": metadata.get("presetName"),
        "product_version": metadata.get("productVersion"),
        "tags": metadata.get("tags"),
        "payload_version": preset.payload_version,
        "metadata_length": preset.metadata_length,
        "cbor_length": preset.cbor_length,
        "compressed_length": preset.compressed_length,
        "top_level_keys": len(data_keys),
        "top_level_key_sha256": hashlib.sha256(
            "\n".join(data_keys).encode("utf-8")
        ).hexdigest(),
        "recursive_types": dict(type_counts),
        "plain_param_dicts": plain_param_dicts,
        "plain_param_values": plain_param_values,
        "plain_param_defaults": plain_param_defaults,
        "active_mod_slots": active_mod_slots,
        "binary_fields": binary_fields,
        "embedded_fields": embedded_fields,
        "path_references": path_references,
        "nonempty_note_lists": note_lists,
    }


def summarize(preset: Serum2Preset) -> dict[str, Any]:
    inventory = Inventory()
    inventory_value(preset.data, inventory)
    metadata_keys = list(preset.metadata) if isinstance(preset.metadata, dict) else []
    data_keys = list(preset.data) if isinstance(preset.data, dict) else []
    return {
        "file": str(preset.path),
        "container": {
            "metadata_length": preset.metadata_length,
            "cbor_length": preset.cbor_length,
            "compressed_length": preset.compressed_length,
            "payload_version": preset.payload_version,
        },
        "metadata_type": type(preset.metadata).__name__,
        "metadata_keys": metadata_keys,
        "metadata": preset.metadata,
        "data_type": type(preset.data).__name__,
        "top_level_key_count": len(data_keys),
        "top_level_keys": data_keys,
        "recursive_type_counts": dict(inventory.types),
        "binary_fields": inventory.binary_fields,
        "parameter_fields": compact(inventory.parameter_fields, 40),
        "candidate_state_or_wavetable_fields": compact(inventory.candidate_fields, 40),
        "largest_dicts": compact(sorted(inventory.largest_dicts, reverse=True), 20),
        "largest_lists": compact(sorted(inventory.largest_lists, reverse=True), 20),
        "largest_strings": compact(sorted(inventory.largest_strings, reverse=True), 20),
        "unique_key_count": len(inventory.key_names),
        "most_common_keys": inventory.key_names.most_common(40),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("presets", type=Path, nargs="+")
    args = parser.parse_args()
    if args.compact:
        print(
            json.dumps(
                [compact_summary(parse_serum2_preset(path)) for path in args.presets],
                indent=2,
                default=str,
            )
        )
        return 0
    for path in args.presets:
        print(json.dumps(summarize(parse_serum2_preset(path)), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

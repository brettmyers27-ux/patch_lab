"""Canonical, invertible numeric targets for decoded Serum 2 settings graphs.

The on-disk Serum 2 graph is sparse: ``plainParams`` stores overrides while
omitted fields inherit the plug-in default.  This module keeps a parallel mask
so sparse values are never confused with numeric zero during model training.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 2
MOD_SLOT_COUNT = 64
MACRO_COUNT = 8
FX_RACK_COUNT = 3
ASSET_KEYS = {"relativePathToWT", "relativePathToNoiseSample", "relativePathToIR"}
VARIABLE_KEYS = {
    "clip",
    "curveData",
    "pathData",
    "embeddedIR",
    "embeddedWT",
    "flex",
    "scale",
}
ROOT_EXCLUSIONS = {
    "SerumGUI",
    "Osc",
    "WTOsc",
    "GranularOsc",
    "MultiSampleOsc",
    "SpectralOsc",
}
DISCRETE_TOKENS = (
    "active",
    "beat",
    "bipolar",
    "bypass",
    "clipid",
    "continuous",
    "default",
    "dotted",
    "enable",
    "enabled",
    "invert",
    "launch",
    "link",
    "loop",
    "menu",
    "mode",
    "mono",
    "octave",
    "oversampling",
    "playback",
    "poly",
    "quantize",
    "range",
    "retrig",
    "reverse",
    "shape",
    "stack",
    "sync",
    "thru",
    "toggle",
    "transpose",
    "triplet",
    "type",
    "unison",
    "wrap",
)


def category_for_field(name: str) -> str:
    lower = name.casefold()
    if lower.startswith("modslot"):
        return "mod_matrix"
    if lower.startswith("fxrack"):
        return "fx"
    if lower.startswith("macro"):
        return "macros"
    if lower.startswith("env"):
        return "envelopes"
    if lower.startswith("lfo") or lower.startswith("lfopoint"):
        return "lfos"
    if "filter" in lower:
        return "filters"
    if lower.startswith("oscillator") or "osc" in lower:
        return "oscillators"
    return "global_other"


def _json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _module_identity(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    names = sorted(key for key, value in item.items() if key.startswith("FX") and isinstance(value, dict))
    if not names:
        return None
    return {"name": names[0], "type": item.get("type")}


def audit_configuration(graphs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fx_slots = []
    mod_param_names: set[str] = set()
    for rack_index in range(FX_RACK_COUNT):
        fx_slots.append(
            max(
                (
                    len(graph.get(f"FXRack{rack_index}", {}).get("FX", []))
                    if isinstance(graph.get(f"FXRack{rack_index}"), dict)
                    else 0
                )
                for graph in graphs
            )
        )
    for graph in graphs:
        for slot_index in range(MOD_SLOT_COUNT):
            slot = graph.get(f"ModSlot{slot_index}", {})
            params = slot.get("plainParams") if isinstance(slot, dict) else None
            if isinstance(params, dict):
                mod_param_names.update(str(name) for name in params)
    return {"fx_slots_per_rack": fx_slots, "mod_param_names": sorted(mod_param_names)}


def _extract_plain_component(value: Any, path: str, output: dict[str, Any]) -> None:
    if not isinstance(value, dict):
        return
    params = value.get("plainParams")
    if isinstance(params, dict):
        for name, item in params.items():
            if isinstance(item, (str, int, float, bool)):
                output[f"{path}.plainParams.{name}"] = item
    for key, item in value.items():
        if key == "plainParams" or key in VARIABLE_KEYS:
            continue
        if key in ASSET_KEYS and isinstance(item, str):
            output[f"{path}.{key}"] = item
        elif isinstance(item, dict):
            _extract_plain_component(item, f"{path}.{key}", output)


def extract_target_values(
    graph: Mapping[str, Any], configuration: Mapping[str, Any]
) -> dict[str, Any]:
    """Extract the supported, fixed-cardinality target subset from one graph."""

    output: dict[str, Any] = {}
    mod_param_names = [str(name) for name in configuration["mod_param_names"]]
    for slot_index in range(MOD_SLOT_COUNT):
        slot_name = f"ModSlot{slot_index}"
        raw = graph.get(slot_name, {})
        slot = raw if isinstance(raw, dict) else {}
        source = slot.get("source")
        destination = None
        if any(
            key in slot
            for key in (
                "destModuleTypeString",
                "destModuleID",
                "destModuleParamID",
                "destModuleParamName",
            )
        ):
            destination = {
                "module_type": slot.get("destModuleTypeString"),
                "module_id": slot.get("destModuleID"),
                "param_id": slot.get("destModuleParamID"),
                "param_name": slot.get("destModuleParamName"),
            }
        output[f"{slot_name}.source"] = source
        output[f"{slot_name}.destination"] = destination
        params = slot.get("plainParams")
        params = params if isinstance(params, dict) else {}
        for param_name in mod_param_names:
            output[f"{slot_name}.plainParams.{param_name}"] = float(params.get(param_name, 0.0))

    for macro_index in range(MACRO_COUNT):
        macro_name = f"Macro{macro_index}"
        macro = graph.get(macro_name, {})
        params = macro.get("plainParams") if isinstance(macro, dict) else None
        value = params.get("kParamValue", 0.0) if isinstance(params, dict) else 0.0
        output[f"{macro_name}.plainParams.kParamValue"] = float(value)

    fx_slots = [int(value) for value in configuration["fx_slots_per_rack"]]
    for rack_index, slot_count in enumerate(fx_slots):
        rack_name = f"FXRack{rack_index}"
        rack = graph.get(rack_name, {})
        items = rack.get("FX", []) if isinstance(rack, dict) else []
        items = items if isinstance(items, list) else []
        rack_params = rack.get("plainParams") if isinstance(rack, dict) else None
        if isinstance(rack_params, dict):
            for name, value in rack_params.items():
                if isinstance(value, (str, int, float, bool)):
                    output[f"{rack_name}.plainParams.{name}"] = value
        for slot_index in range(slot_count):
            item = items[slot_index] if slot_index < len(items) else None
            base = f"{rack_name}.slot{slot_index}"
            identity = _module_identity(item)
            output[f"{base}.module"] = identity
            if identity is not None and isinstance(item, dict):
                module = item.get(str(identity["name"]), {})
                _extract_plain_component(module, f"{base}.{identity['name']}", output)

    for key, value in graph.items():
        if (
            key in ROOT_EXCLUSIONS
            or re.fullmatch(r"ModSlot\d+", key)
            or re.fullmatch(r"Macro\d+", key)
            or re.fullmatch(r"FXRack\d+", key)
        ):
            continue
        if isinstance(value, dict):
            _extract_plain_component(value, key, output)
    return output


def _numeric_is_discrete(name: str, values: Sequence[float]) -> bool:
    unique = sorted(set(float(value) for value in values))
    integral = all(math.isclose(value, round(value), abs_tol=1e-8) for value in unique)
    lower = name.casefold().replace("_", "")
    named_discrete = any(token in lower for token in DISCRETE_TOKENS)
    return integral and len(unique) <= 32 and (named_discrete or len(unique) <= 8)


def build_schema_and_arrays(
    preset_ids: Sequence[int], graphs: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if len(preset_ids) != len(graphs):
        raise ValueError("preset_ids and graphs must have identical lengths")
    configuration = audit_configuration(graphs)
    extracted = [extract_target_values(graph, configuration) for graph in graphs]
    field_names = sorted({name for row in extracted for name in row})
    values_by_field: dict[str, list[Any]] = defaultdict(list)
    for row in extracted:
        for name, value in row.items():
            values_by_field[name].append(value)

    fields: list[dict[str, Any]] = []
    vector_offset = 0
    for name in field_names:
        values = values_by_field[name]
        forced_category = name.endswith((".source", ".destination", ".module")) or any(
            name.endswith(f".{key}") for key in ASSET_KEYS
        )
        categorical = forced_category or any(not isinstance(value, (int, float, bool)) for value in values)
        if not categorical:
            categorical = _numeric_is_discrete(name, [float(value) for value in values])
        common = {
            "name": name,
            "category": category_for_field(name),
            "present_count": sum(name in row for row in extracted),
        }
        if categorical:
            category_map = {_json_key(value): value for value in values}
            categories = [category_map[key] for key in sorted(category_map)]
            fields.append(
                {
                    **common,
                    "index": vector_offset,
                    "width": len(categories),
                    "encoding": "one_hot",
                    "categories": categories,
                    "stepped": True,
                }
            )
            vector_offset += len(categories)
        else:
            numeric = [float(value) for value in values]
            fields.append(
                {
                    **common,
                    "index": vector_offset,
                    "width": 1,
                    "encoding": "minmax_float",
                    "minimum": min(numeric),
                    "maximum": max(numeric),
                    "stepped": False,
                }
            )
            vector_offset += 1

    vectors = np.zeros((len(graphs), vector_offset), dtype=np.float32)
    masks = np.zeros_like(vectors, dtype=np.bool_)
    for row_index, row in enumerate(extracted):
        for field in fields:
            name = str(field["name"])
            if name not in row:
                continue
            index = int(field["index"])
            value = row[name]
            if field["encoding"] == "one_hot":
                categories = field["categories"]
                lookup = {_json_key(category): position for position, category in enumerate(categories)}
                position = lookup[_json_key(value)]
                vectors[row_index, index + position] = 1.0
                masks[row_index, index : index + len(categories)] = True
            else:
                minimum, maximum = float(field["minimum"]), float(field["maximum"])
                vectors[row_index, index] = (
                    (float(value) - minimum) / (maximum - minimum) if maximum > minimum else 0.0
                )
                masks[row_index, index] = True

    field_breakdown = Counter(str(field["category"]) for field in fields)
    dimension_breakdown = Counter()
    for field in fields:
        dimension_breakdown[str(field["category"])] += int(field["width"])
    encoding_breakdown = Counter(str(field["encoding"]) for field in fields)
    schema = {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Sparse Serum 2 plainParams plus fixed 64-slot modulation routing, "
            "three fixed-index FX racks, eight macro values, and resolvable factory asset paths."
        ),
        "vector_length": vector_offset,
        "conceptual_field_count": len(fields),
        "preset_count": len(graphs),
        "configuration": configuration,
        "categorical_encoding": (
            "Stable JSON-sorted one-hot slices; inference takes argmax and decodes through the saved table."
        ),
        "numeric_encoding": "Per-field corpus min/max scaling to [0,1].",
        "missing_values": "Stored as vector zero with a parallel false mask; masked SmoothL1 ignores them.",
        "included": [
            "Scalar numeric/enum values in component plainParams",
            "64 modulation sources, destinations, and scalar slot parameters",
            "Fixed-index FX module identity and scalar module parameters",
            "Eight macro values; macro assignments are represented by modulation sources",
            "Resolvable wavetable, noise-sample, and convolution-IR paths",
        ],
        "excluded": [
            "Embedded wavetable/IR sample arrays",
            "Arp and MIDI clip note sequences (their scalar plainParams remain included)",
            "Variable LFO curve/path point arrays and point-modulation arrays",
            "GUI/editor state and user-facing labels",
            "Variable multisample zone/sample content",
        ],
        "field_breakdown": dict(sorted(field_breakdown.items())),
        "vector_dimension_breakdown": dict(sorted(dimension_breakdown.items())),
        "encoding_breakdown": dict(sorted(encoding_breakdown.items())),
        "fields": fields,
    }
    return schema, vectors, masks


def decode_vector(
    vector: np.ndarray,
    schema: Mapping[str, Any],
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Decode a target vector into a sparse settings graph usable by reconstruction."""

    vector = np.asarray(vector).reshape(-1)
    if vector.shape[0] != int(schema["vector_length"]):
        raise ValueError("Vector length does not match Serum 2 schema")
    if mask is not None:
        mask = np.asarray(mask, dtype=np.bool_).reshape(-1)
        if mask.shape != vector.shape:
            raise ValueError("Mask shape does not match vector")

    decoded: dict[str, Any] = {}
    for field in schema["fields"]:
        index = int(field["index"])
        if mask is not None and not bool(mask[index]):
            continue
        value = float(np.clip(vector[index], 0.0, 1.0))
        if field["encoding"] == "one_hot":
            categories = field["categories"]
            width = int(field["width"])
            position = int(np.argmax(vector[index : index + width]))
            item = copy.deepcopy(categories[position])
        else:
            minimum, maximum = float(field["minimum"]), float(field["maximum"])
            item = minimum + value * (maximum - minimum)
        decoded[str(field["name"])] = item

    graph: dict[str, Any] = {}
    configuration = schema["configuration"]
    mod_param_names = set(configuration["mod_param_names"])
    for slot_index in range(MOD_SLOT_COUNT):
        name = f"ModSlot{slot_index}"
        source = decoded.get(f"{name}.source")
        destination = decoded.get(f"{name}.destination")
        params = {
            param: decoded[f"{name}.plainParams.{param}"]
            for param in mod_param_names
            if f"{name}.plainParams.{param}" in decoded
            and not math.isclose(float(decoded[f"{name}.plainParams.{param}"]), 0.0, abs_tol=1e-12)
        }
        slot: dict[str, Any] = {"plainParams": params if params else "default"}
        if source is not None:
            slot["source"] = source
        if isinstance(destination, dict):
            slot.update(
                {
                    "destModuleTypeString": destination.get("module_type"),
                    "destModuleID": destination.get("module_id"),
                    "destModuleParamID": destination.get("param_id"),
                    "destModuleParamName": destination.get("param_name"),
                }
            )
        graph[name] = slot

    for macro_index in range(MACRO_COUNT):
        name = f"Macro{macro_index}"
        key = f"{name}.plainParams.kParamValue"
        value = decoded.get(key, 0.0)
        graph[name] = {"plainParams": {"kParamValue": value} if not math.isclose(float(value), 0.0) else "default"}

    for rack_index, slot_count in enumerate(configuration["fx_slots_per_rack"]):
        rack_name = f"FXRack{rack_index}"
        items: list[dict[str, Any]] = []
        for slot_index in range(int(slot_count)):
            base = f"{rack_name}.slot{slot_index}"
            identity = decoded.get(f"{base}.module")
            if not isinstance(identity, dict) or not identity.get("name"):
                continue
            module_name = str(identity["name"])
            module: dict[str, Any] = {"plainParams": {}}
            prefix = f"{base}.{module_name}."
            for name, value in decoded.items():
                if not name.startswith(prefix):
                    continue
                suffix = name[len(prefix) :]
                if suffix.startswith("plainParams."):
                    module["plainParams"][suffix.removeprefix("plainParams.")] = value
                elif suffix in ASSET_KEYS:
                    module[suffix] = value
            if not module["plainParams"]:
                module["plainParams"] = "default"
            items.append({module_name: module, "type": identity.get("type")})
        graph[rack_name] = {"FX": items, "plainParams": "default", "displayName": ""}

    special_prefixes = ("ModSlot", "Macro", "FXRack")
    for name, value in decoded.items():
        if name.startswith(special_prefixes):
            continue
        parts = name.split(".")
        cursor: dict[str, Any] = graph
        for part in parts[:-1]:
            child = cursor.get(part)
            if not isinstance(child, dict):
                child = {}
                cursor[part] = child
            cursor = child
        cursor[parts[-1]] = value
    return graph


def encode_graph_with_schema(
    graph: Mapping[str, Any], schema: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Encode one new graph against the shipped schema without expanding it.

    Values that were never present in the factory corpus are left masked rather
    than being silently forced into an unrelated category.
    """

    extracted = extract_target_values(graph, schema["configuration"])
    vector = np.zeros(int(schema["vector_length"]), dtype=np.float32)
    mask = np.zeros_like(vector, dtype=np.bool_)
    matched = 0
    unknown_categories = 0
    for field in schema["fields"]:
        name = str(field["name"])
        if name not in extracted:
            continue
        index = int(field["index"])
        if field["encoding"] == "one_hot":
            lookup = {
                _json_key(category): offset
                for offset, category in enumerate(field["categories"])
            }
            offset = lookup.get(_json_key(extracted[name]))
            if offset is None:
                unknown_categories += 1
                continue
            width = int(field["width"])
            vector[index + offset] = 1.0
            mask[index : index + width] = True
        else:
            minimum = float(field["minimum"])
            maximum = float(field["maximum"])
            value = float(extracted[name])
            vector[index] = (
                np.clip((value - minimum) / (maximum - minimum), 0.0, 1.0)
                if maximum > minimum
                else 0.0
            )
            mask[index] = True
        matched += 1
    return vector, mask, {
        "source_values": len(extracted),
        "matched_fields": matched,
        "unknown_categories": unknown_categories,
    }


def validate_round_trip(
    graphs: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
    vectors: np.ndarray,
    masks: np.ndarray,
) -> dict[str, Any]:
    configuration = schema["configuration"]
    numeric_max_error = 0.0
    categorical_errors = 0
    compared = 0
    for row_index, graph in enumerate(graphs):
        original = extract_target_values(graph, configuration)
        rebuilt = extract_target_values(decode_vector(vectors[row_index], schema, masks[row_index]), configuration)
        for name, expected in original.items():
            if name not in rebuilt:
                categorical_errors += 1
                continue
            actual = rebuilt[name]
            compared += 1
            if isinstance(expected, (int, float, bool)) and isinstance(actual, (int, float, bool)):
                numeric_max_error = max(numeric_max_error, abs(float(expected) - float(actual)))
            elif _json_key(expected) != _json_key(actual):
                categorical_errors += 1
    return {
        "compared_values": compared,
        "categorical_errors": categorical_errors,
        "numeric_max_abs_error": numeric_max_error,
        "pass": categorical_errors == 0 and numeric_max_error < 1e-3,
    }


def save_schema(path: Path, schema: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")


def expanded_output_mapping(schema: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return one reporting/inference descriptor per numeric output dimension."""

    result: list[dict[str, Any]] = []
    for field in schema["fields"]:
        start = int(field["index"])
        width = int(field.get("width", 1))
        if field["encoding"] == "one_hot":
            for offset, category in enumerate(field["categories"]):
                result.append(
                    {
                        "index": start + offset,
                        "name": f"{field['name']} == {_json_key(category)}",
                        "field_name": field["name"],
                        "category": field["category"],
                        "encoding": "one_hot",
                        "category_value": category,
                        "stepped": True,
                    }
                )
        else:
            if width != 1:
                raise ValueError(f"Numeric field has unexpected width: {field['name']}")
            result.append(
                {
                    "index": start,
                    "name": field["name"],
                    "field_name": field["name"],
                    "category": field["category"],
                    "encoding": field["encoding"],
                    "stepped": False,
                }
            )
    if len(result) != int(schema["vector_length"]):
        raise ValueError("Expanded mapping length does not match Serum 2 vector")
    return result

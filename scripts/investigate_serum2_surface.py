#!/usr/bin/env python3
"""Audit Serum 2's live FX and modulation automation surface."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.param_calibration import CalibrationTable, canonical_display
from core.platform_env import ENV
from core.plugin_host import make_dawdreamer_processor


DEFAULT_CALIBRATION = PROJECT_ROOT / "data" / "models" / "serum2_param_calibration.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "models" / "serum2_surface_investigation.json"
FX_TERMS = ("fx", "effect", "insert", "slot")
MOD_TERMS = ("mod", "matrix", "route", "src", "source", "dest", "assign")


def _contains(name: str, terms: tuple[str, ...]) -> bool:
    lowered = name.casefold()
    return any(term in lowered for term in terms)


def _description_row(description: Any, fallback_index: int) -> dict[str, Any]:
    if not isinstance(description, dict):
        return {"index": fallback_index, "name": str(description)}
    keep = (
        "index",
        "name",
        "numSteps",
        "isBoolean",
        "isDiscrete",
        "label",
        "category",
        "isAutomatable",
        "defaultValueText",
        "min",
        "max",
        "valueStrings",
    )
    row = {key: description.get(key) for key in keep}
    row["index"] = int(description.get("index", fallback_index))
    row["name"] = str(description.get("name", description.get("label", fallback_index)))
    return row


def _fx_signature(descriptions: list[Any], fx_indices: list[int]) -> dict[int, tuple[str, ...]]:
    result: dict[int, tuple[str, ...]] = {}
    for index in fx_indices:
        item = descriptions[index]
        if isinstance(item, dict):
            result[index] = tuple(
                str(item.get(key, "")) for key in ("name", "label", "min", "max")
            )
        else:
            result[index] = (str(item), "", "", "")
    return result


def _representative_samples(entry: dict[str, Any], maximum: int = 256) -> list[tuple[float, str]]:
    by_display: dict[str, tuple[float, str]] = {}
    for normalized, display in entry.get("samples", []):
        key = canonical_display(display)
        by_display.setdefault(key, (float(normalized), str(display)))
    values = sorted(by_display.values())
    if len(values) <= maximum:
        return values
    # A hidden selector reported as continuous should still be exercised broadly.
    positions = {round(i * (len(values) - 1) / (maximum - 1)) for i in range(maximum)}
    return [values[position] for position in sorted(positions)]


def _looks_like_parameter(display: str, parameter_names: list[str]) -> list[str]:
    wanted = set(re.findall(r"[a-z]+|\d+", display.casefold()))
    if not wanted or canonical_display(display) in {"--", "on", "off"}:
        return []
    matches: list[str] = []
    for name in parameter_names:
        tokens = set(re.findall(r"[a-z]+|\d+", name.casefold()))
        if len(wanted) >= 2 and wanted == tokens:
            matches.append(name)
    return matches[:10]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def investigate(calibration_path: Path) -> dict[str, Any]:
    calibration = CalibrationTable.load(calibration_path)
    entries = {
        int(entry["index"]): entry
        for group in calibration.payload["parameters"].values()
        for entry in group
    }
    candidates = [item for item in ENV.plugins_for("serum2") if item.format == "VST3"]
    if not candidates:
        raise RuntimeError("No Serum 2 VST3 candidate is available")
    _engine, processor = make_dawdreamer_processor(candidates[0])
    raw_descriptions = list(processor.get_parameters_description())
    descriptions = [_description_row(item, index) for index, item in enumerate(raw_descriptions)]
    parameter_names = [str(item["name"]) for item in descriptions]
    fx_rows = [item for item in descriptions if _contains(str(item["name"]), FX_TERMS)]
    mod_rows = [item for item in descriptions if _contains(str(item["name"]), MOD_TERMS)]
    fx_indices = [int(item["index"]) for item in fx_rows]
    fx_selectors = [
        item
        for item in fx_rows
        if any(token in str(item["name"]).casefold() for token in ("type", "mode", "select"))
        and (item.get("isDiscrete") or int(item.get("numSteps") or 0) <= 256)
    ]

    original = {index: float(processor.get_parameter(index)) for index in range(len(descriptions))}
    baseline = _fx_signature(raw_descriptions, fx_indices)
    dynamic_changes: list[dict[str, Any]] = []
    swept_candidates: list[dict[str, Any]] = []
    try:
        # No FX-named selector exists. Sweep every bounded enum/stepped control on the
        # core 0..540 surface as a defensive check for a hidden topology selector.
        for index in range(min(541, len(descriptions))):
            description = descriptions[index]
            entry = entries.get(index)
            if entry is None:
                continue
            num_steps = int(description.get("numSteps") or 0)
            is_candidate = bool(description.get("isDiscrete")) or entry.get("kind") == "stepped"
            is_candidate = is_candidate or (1 < int(entry.get("unique_displays", 0)) <= 128)
            if not is_candidate or num_steps > 256 and entry.get("kind") == "continuous":
                continue
            samples = _representative_samples(entry)
            swept_candidates.append(
                {"index": index, "name": description["name"], "enum_values_tested": len(samples)}
            )
            for normalized, display in samples:
                processor.set_parameter(index, normalized)
                changed = _fx_signature(list(processor.get_parameters_description()), fx_indices)
                differences = []
                for fx_index in fx_indices:
                    if changed[fx_index] != baseline[fx_index]:
                        differences.append(
                            {
                                "fx_index": fx_index,
                                "before": baseline[fx_index],
                                "after": changed[fx_index],
                            }
                        )
                if differences:
                    dynamic_changes.append(
                        {
                            "selector_index": index,
                            "selector_name": description["name"],
                            "normalized": normalized,
                            "display": display,
                            "differences": differences,
                        }
                    )
            processor.set_parameter(index, original[index])
    finally:
        for index, value in original.items():
            processor.set_parameter(index, value)

    mod_enums: list[dict[str, Any]] = []
    route_like_displays: list[dict[str, Any]] = []
    for row in mod_rows:
        index = int(row["index"])
        entry = entries.get(index)
        if entry is None:
            continue
        discrete = bool(row.get("isDiscrete")) or entry.get("kind") == "stepped"
        if not discrete:
            continue
        displays = sorted({str(sample[1]) for sample in entry.get("samples", [])})
        mod_enums.append({"index": index, "name": row["name"], "displays": displays})
        for display in displays:
            matches = _looks_like_parameter(display, parameter_names)
            if matches:
                route_like_displays.append(
                    {"index": index, "name": row["name"], "display": display, "matches": matches}
                )

    return {
        "plugin": str(candidates[0].path),
        "parameter_count": len(descriptions),
        "all_parameter_names": [
            {"index": int(item["index"]), "name": item["name"]} for item in descriptions
        ],
        "fx": {
            "terms": list(FX_TERMS),
            "matches": fx_rows,
            "named_type_selectors": fx_selectors,
            "defensive_enum_sweep": swept_candidates,
            "dynamic_name_or_label_changes": dynamic_changes,
        },
        "modulation": {
            "terms": list(MOD_TERMS),
            "matches": mod_rows,
            "stepped_or_enum_matches": mod_enums,
            "route_like_enum_displays": route_like_displays,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = investigate(args.calibration)
    write_json(args.output, result)
    for section in ("fx", "modulation"):
        print(f"=== {section.upper()} PARAMETER MATCHES ===")
        for item in result[section]["matches"]:
            print(f"{item['index']:4d}  {item['name']}")
    print("=== CONCLUSIONS ===")
    print(f"FX_NAMED_TYPE_SELECTORS={len(result['fx']['named_type_selectors'])}")
    print(f"FX_ENUM_CONTROLS_SWEPT={len(result['fx']['defensive_enum_sweep'])}")
    print(f"FX_DYNAMIC_RELABEL_EVENTS={len(result['fx']['dynamic_name_or_label_changes'])}")
    print(f"MOD_STEPPED_ENUM_MATCHES={len(result['modulation']['stepped_or_enum_matches'])}")
    print(f"MOD_ROUTE_LIKE_ENUM_DISPLAYS={len(result['modulation']['route_like_enum_displays'])}")
    print(f"OUTPUT={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Inspect Serum 2's host-visible VST3 program-list surface."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.platform_env import ENV
from core.plugin_host import (
    changed_parameter_count,
    dawdreamer_parameter_display,
    dump_dawdreamer_parameters,
    make_dawdreamer_processor,
)


OUTPUT = PROJECT_ROOT / "data" / "models" / "serum2_native_program_report.json"


def relevant_api(obj: Any) -> list[dict[str, Any]]:
    result = []
    for name in sorted(dir(obj)):
        if not any(token in name.lower() for token in ("program", "unit", "preset", "state")):
            continue
        value = getattr(obj, name)
        item: dict[str, Any] = {"name": name, "callable": callable(value)}
        if callable(value):
            try:
                item["signature"] = str(inspect.signature(value))
            except (TypeError, ValueError):
                item["signature"] = None
            item["doc"] = inspect.getdoc(value)
        else:
            item["type"] = type(value).__name__
        result.append(item)
    return result


def main() -> int:
    candidate = next(item for item in ENV.plugins_for("serum2") if item.format == "VST3")
    engine, daw = make_dawdreamer_processor(candidate)
    initial = dump_dawdreamer_parameters(daw)
    labels = []
    for index in range(128):
        value = index / 128.0
        daw.set_parameter(540, value)
        labels.append(dawdreamer_parameter_display(daw, 540))
    daw.set_parameter(540, 0.5)
    after = dump_dawdreamer_parameters(daw)

    from pedalboard import load_plugin

    pedalboard = load_plugin(str(candidate.path), plugin_name="Serum 2")
    payload = {
        "plugin": str(candidate.path),
        "dawdreamer_api": relevant_api(daw),
        "pedalboard_api": relevant_api(pedalboard),
        "program_api_names": {
            "dawdreamer": [name for name in dir(daw) if "program" in name.lower() or "unit" in name.lower()],
            "pedalboard": [
                name for name in dir(pedalboard) if "program" in name.lower() or "unit" in name.lower()
            ],
        },
        "bank_parameter": {
            "index": 540,
            "labels": labels,
            "unique_labels": len(set(labels)),
            "first": labels[0],
            "last": labels[-1],
            "changed_parameters_total": changed_parameter_count(initial, after),
            "changed_parameters_excluding_bank": sum(
                abs(before.norm_value - loaded.norm_value) > 1e-4
                for before, loaded in zip(initial, after)
                if before.index != 540
            ),
            "maps_to_preset_folder": False,
        },
        "usable_program_list": False,
        "five_fixture_test": "not applicable: no names or selector map to preset-folder contents",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"usable_program_list": False, "bank_parameter": payload["bank_parameter"]}, indent=2))
    print(f"OUTPUT={OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

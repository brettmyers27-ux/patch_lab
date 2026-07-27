#!/usr/bin/env python3
"""Step 2c: reconstruct Serum 2 presets through live automation and verify them."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.param_calibration import CalibrationTable
from core.platform_env import ENV
from core.serum2_apply import apply_preset
from core.serum2_preset import parse_serum2_preset


DEFAULT_CALIBRATION = PROJECT_ROOT / "data" / "models" / "serum2_param_calibration.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "models" / "serum2_step2c_report.json"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def vectors_differ(left: list[float], right: list[float], threshold: float = 1e-4) -> bool:
    return any(abs(a - b) > threshold for a, b in zip(left, right))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("presets", nargs="+", type=Path)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-refine", action="store_true")
    args = parser.parse_args()
    candidates = [item for item in ENV.plugins_for("serum2") if item.format == "VST3"]
    if not candidates:
        raise RuntimeError("No Serum 2 VST3 candidate is available")
    calibration = CalibrationTable.load(args.calibration)
    reports = []
    for path in args.presets:
        print(f"APPLYING={path}", flush=True)
        reports.append(
            apply_preset(
                parse_serum2_preset(path),
                calibration,
                candidates[0],
                refine=not args.no_refine,
            )
        )
    pairwise = []
    for left_index, left in enumerate(reports):
        for right in reports[left_index + 1 :]:
            pairwise.append(
                {
                    "left": left["preset_name"],
                    "right": right["preset_name"],
                    "different": vectors_differ(left["parameter_vector"], right["parameter_vector"]),
                }
            )
    categories: dict[str, Counter[str]] = defaultdict(Counter)
    for report in reports:
        for category, counts in report["category_counts"].items():
            categories[category].update(counts)
    total = sum(int(report["plain_param_total"]) for report in reports)
    applied = sum(int(report["applied"]) for report in reports)
    gate_pass = (
        all(report["section_6_1_init_pass"] for report in reports)
        and all(report["section_6_1_silence_pass"] for report in reports)
        and all(item["different"] for item in pairwise)
        and all(not report["errors"] for report in reports)
    )
    aggregate = {
        "presets": len(reports),
        "plain_param_total": total,
        "applied": applied,
        "application_coverage": applied / total if total else 0.0,
        "category_counts": {key: dict(value) for key, value in categories.items()},
        "pairwise_vectors_different": all(item["different"] for item in pairwise),
        "section_6_1_pass": gate_pass,
    }
    payload = {"aggregate": aggregate, "pairwise": pairwise, "presets": reports}
    write_json(args.output, payload)
    compact = {
        "aggregate": aggregate,
        "presets": [
            {
                key: report[key]
                for key in (
                    "preset_name",
                    "plain_param_total",
                    "mapped",
                    "applied",
                    "application_coverage",
                    "category_counts",
                    "changed_from_init",
                    "rms_dbfs",
                    "peak_dbfs",
                    "errors",
                    "section_6_1_init_pass",
                    "section_6_1_silence_pass",
                )
            }
            for report in reports
        ],
        "output": str(args.output),
    }
    print(json.dumps(compact, indent=2))
    print(f"STEP_2C_GATE={'PASS' if gate_pass else 'FAIL'}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

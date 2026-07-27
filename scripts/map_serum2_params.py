#!/usr/bin/env python3
"""Build and print the Step 2b Serum 2 mapping coverage report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.param_calibration import CalibrationTable
from core.serum2_mapping import Serum2Mapper, aggregate_reports
from core.serum2_preset import parse_serum2_preset


DEFAULT_CALIBRATION = PROJECT_ROOT / "data" / "models" / "serum2_param_calibration.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "models" / "serum2_mapping_report.json"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("presets", type=Path, nargs="+")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    mapper = Serum2Mapper(CalibrationTable.load(args.calibration))
    reports = [mapper.map_preset(parse_serum2_preset(path)) for path in args.presets]
    payload = {"aggregate": aggregate_reports(reports), "presets": reports}
    write_json(args.output, payload)
    compact = {
        "aggregate": payload["aggregate"],
        "presets": [
            {
                key: report[key]
                for key in (
                    "preset_name",
                    "plain_param_total",
                    "mapped",
                    "mapping_coverage",
                    "status_counts",
                    "category_counts",
                )
            }
            for report in reports
        ],
        "output": str(args.output),
    }
    print(json.dumps(compact, indent=2))
    print("STEP_2B_GATE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build and verify fixed-length Serum 2 training targets from SQLite."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.db import DEFAULT_DB_PATH
from core.serum2_targets import build_schema_and_arrays, save_schema, validate_round_trip


DEFAULT_SCHEMA = PROJECT_ROOT / "data" / "models" / "serum2_target_schema.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data" / "features" / "serum2_targets.npz"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "models" / "serum2_target_report.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    connection = sqlite3.connect(args.db)
    rows = connection.execute(
        "SELECT preset_id,settings_json FROM serum2_full_settings ORDER BY preset_id"
    ).fetchall()
    preset_ids = [int(row[0]) for row in rows]
    graphs = [json.loads(row[1]) for row in rows]
    if len(graphs) != 710:
        raise RuntimeError(f"Expected 710 Serum 2 settings graphs, found {len(graphs)}")

    schema, vectors, masks = build_schema_and_arrays(preset_ids, graphs)
    round_trip = validate_round_trip(graphs, schema, vectors, masks)
    if not round_trip["pass"]:
        raise RuntimeError(f"Serum 2 target round-trip failed: {round_trip}")

    save_schema(args.schema, schema)
    args.targets.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.targets,
        preset_ids=np.asarray(preset_ids, dtype=np.int64),
        vectors=vectors,
        masks=masks,
    )
    report = {
        "schema_path": str(args.schema.resolve()),
        "targets_path": str(args.targets.resolve()),
        "preset_count": len(preset_ids),
        "vector_length": int(schema["vector_length"]),
        "conceptual_field_count": int(schema["conceptual_field_count"]),
        "field_breakdown": schema["field_breakdown"],
        "vector_dimension_breakdown": schema["vector_dimension_breakdown"],
        "encoding_breakdown": schema["encoding_breakdown"],
        "fx_slots_per_rack": schema["configuration"]["fx_slots_per_rack"],
        "mod_param_names": schema["configuration"]["mod_param_names"],
        "mask_density": float(masks.mean()),
        "round_trip": round_trip,
        "excluded": schema["excluded"],
        "gate_pass": bool(round_trip["pass"]),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

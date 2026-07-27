#!/usr/bin/env python3
"""Step 3: reconstruct and catalog every Serum 2 preset in SQLite."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.db import DEFAULT_DB_PATH, Database, PresetRecord
from core.param_calibration import CalibrationTable
from core.platform_env import ENV
from core.plugin_host import dump_dawdreamer_parameters, make_dawdreamer_processor
from core.serum2_apply import apply_preset_on_processor
from core.serum2_preset import parse_serum2_preset


DEFAULT_CALIBRATION = PROJECT_ROOT / "data" / "models" / "serum2_param_calibration.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "models" / "serum2_catalog_report.json"
STRATEGY = "VST3/S2-cbor-parameter-reconstruction-v1"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def load_reports(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    reports = payload.get("presets", {})
    return reports if isinstance(reports, dict) else {}


def serum2_records(database: Database) -> list[PresetRecord]:
    with database.connect() as connection:
        rows = connection.execute(
            "SELECT * FROM presets WHERE synth='serum2' ORDER BY id"
        ).fetchall()
    return [database._preset(row) for row in rows]


def compact_report(preset_id: int, report: dict[str, Any], status: str) -> dict[str, Any]:
    keep = (
        "file",
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
    result = {key: report[key] for key in keep}
    result.update({"preset_id": preset_id, "status": status})
    return result


def aggregate(database: Database, reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    with database.connect() as connection:
        status_rows = connection.execute(
            "SELECT status,COUNT(*) AS count FROM presets WHERE synth='serum2' GROUP BY status"
        ).fetchall()
        cardinality = connection.execute(
            "SELECT COUNT(*) AS presets,MIN(n) AS minimum,MAX(n) AS maximum FROM "
            "(SELECT p.id,COUNT(pa.param_index) AS n FROM presets p "
            "LEFT JOIN params pa ON pa.preset_id=p.id "
            "WHERE p.synth='serum2' AND p.status='params_dumped' GROUP BY p.id)"
        ).fetchone()
    statuses = {str(row["status"]): int(row["count"]) for row in status_rows}
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    coverages: list[float] = []
    clipped = 0
    for report in reports.values():
        if "application_coverage" in report:
            coverages.append(float(report["application_coverage"]))
        if float(report.get("peak_dbfs", -999.0)) > 0.0:
            clipped += 1
        for category, counts in report.get("category_counts", {}).items():
            category_counts[category].update(counts)
    total = sum(statuses.values())
    failures = statuses.get("failed_load", 0) + statuses.get("failed_silent", 0)
    return {
        "catalog_total": total,
        "status_counts": statuses,
        "success": statuses.get("params_dumped", 0),
        "failures": failures,
        "failure_rate": failures / total if total else 1.0,
        "reports_recorded": len(reports),
        "mean_application_coverage": sum(coverages) / len(coverages) if coverages else 0.0,
        "category_counts": {key: dict(value) for key, value in category_counts.items()},
        "peak_over_0_dbfs": clipped,
        "parameter_cardinality": {
            "presets": int(cardinality["presets"] or 0),
            "minimum": int(cardinality["minimum"] or 0),
            "maximum": int(cardinality["maximum"] or 0),
        },
    }


def persist(path: Path, database: Database, reports: dict[str, dict[str, Any]]) -> None:
    write_json(path, {"aggregate": aggregate(database, reports), "presets": reports})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    database = Database(args.db)
    records = serum2_records(database)
    if len(records) != 710:
        raise RuntimeError(f"Expected the accepted 710-preset catalog, found {len(records)}")
    candidates = [item for item in ENV.plugins_for("serum2") if item.format == "VST3"]
    if not candidates:
        raise RuntimeError("No Serum 2 VST3 candidate is available")
    calibration = CalibrationTable.load(args.calibration)
    engine, processor = make_dawdreamer_processor(candidates[0])
    initial = dump_dawdreamer_parameters(processor)
    reports = load_reports(args.output)
    pending = [
        record
        for record in records
        if args.force or record.status != "params_dumped" or record.load_strategy != STRATEGY
    ]
    if args.limit is not None:
        pending = pending[: args.limit]
    started = time.monotonic()
    for position, record in enumerate(pending, start=1):
        status = "failed_load"
        report: dict[str, Any] | None = None
        try:
            parsed = parse_serum2_preset(record.path)
            report, parameters = apply_preset_on_processor(
                parsed,
                calibration,
                engine,
                processor,
                initial,
                refine=not args.no_refine,
            )
            if report["errors"]:
                raise RuntimeError(f"{len(report['errors'])} parameter application errors")
            if not report["section_6_1_init_pass"]:
                raise RuntimeError(
                    f"only {report['changed_from_init']} parameters changed from init"
                )
            if not report["section_6_1_silence_pass"]:
                status = "failed_silent"
                raise RuntimeError(f"C4 render is silent at {report['rms_dbfs']:.2f} dBFS")
            database.replace_params(record.id, parameters, STRATEGY)
            status = "params_dumped"
        except Exception as exc:
            database.mark_failed(record.id, status, repr(exc))
            if report is None:
                report = {
                    "file": str(record.path),
                    "preset_name": record.name,
                    "plain_param_total": 0,
                    "mapped": 0,
                    "applied": 0,
                    "application_coverage": 0.0,
                    "category_counts": {},
                    "changed_from_init": 0,
                    "rms_dbfs": float("-inf"),
                    "peak_dbfs": float("-inf"),
                    "errors": [],
                    "section_6_1_init_pass": False,
                    "section_6_1_silence_pass": status != "failed_silent",
                }
            report["errors"] = [*report.get("errors", []), {"error": repr(exc)}]
        reports[str(record.id)] = compact_report(record.id, report, status)
        persist(args.output, database, reports)
        elapsed = max(time.monotonic() - started, 1e-9)
        rate = position / elapsed
        eta = (len(pending) - position) / rate if rate else 0.0
        warning = " PEAK_WARNING" if float(report["peak_dbfs"]) > 0.0 else ""
        print(
            f"SERUM2_PROGRESS={position}/{len(pending)} id={record.id} status={status} "
            f"coverage={float(report['application_coverage']):.2%} "
            f"rms={float(report['rms_dbfs']):.2f}dBFS eta={eta:.1f}s{warning}",
            flush=True,
        )
    final = aggregate(database, reports)
    write_json(args.output, {"aggregate": final, "presets": reports})
    print("SERUM2_CATALOG_SUMMARY=" + json.dumps(final, sort_keys=True))
    passed = final["reports_recorded"] == 710 and final["success"] + final["failures"] == 710
    print(f"STEP_3_GATE={'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

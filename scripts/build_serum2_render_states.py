#!/usr/bin/env python3
"""Build id-only, partitioned VST3 state files for Serum 2 rendering."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.db import DEFAULT_DB_PATH, Database
from core.platform_env import ENV
from core.serum2_preset import Serum2Preset
from core.serum2_state_reconstruct import decode_host_template, reconstruct_vstpreset


DEFAULT_STATE_DIR = PROJECT_ROOT / "data" / "models" / "serum2_render_states"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "models" / "serum2_render_state_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    database = Database(args.db)
    candidate = next(item for item in ENV.plugins_for("serum2") if item.format == "VST3")

    from pedalboard import load_plugin

    live = load_plugin(str(candidate.path), plugin_name="Serum 2")
    template = decode_host_template(bytes(live.preset_data))
    with database.connect() as connection:
        rows = connection.execute(
            """
            SELECT p.id,p.path,f.metadata_json,f.settings_json,f.settings_sha256,
                   f.payload_version,f.cbor_length,f.compressed_length
            FROM presets p
            JOIN serum2_full_settings f ON f.preset_id=p.id
            WHERE p.synth='serum2'
            ORDER BY p.id
            """
        ).fetchall()

    args.state_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    failures = []
    unmatched = Counter()
    total_bytes = 0
    for index, row in enumerate(rows, start=1):
        try:
            decoded = Serum2Preset(
                path=Path(row["path"]),
                metadata=json.loads(row["metadata_json"]),
                data=json.loads(row["settings_json"]),
                metadata_length=len(row["metadata_json"].encode("utf-8")),
                cbor_length=int(row["cbor_length"]),
                payload_version=int(row["payload_version"]),
                compressed_length=int(row["compressed_length"]),
            )
            container, partition = reconstruct_vstpreset(decoded, template)
            output = args.state_dir / f"{int(row['id'])}.vstpreset"
            output.write_bytes(container)
            total_bytes += len(container)
            unmatched.update(partition.unmatched_paths)
            reports.append(
                {
                    "preset_id": int(row["id"]),
                    "source": str(row["path"]),
                    "settings_sha256": str(row["settings_sha256"]),
                    "state_path": str(output),
                    "state_bytes": len(container),
                    "structural_coverage": partition.coverage,
                    "unmatched_paths": list(partition.unmatched_paths),
                }
            )
        except Exception as exc:
            failures.append({"preset_id": int(row["id"]), "path": row["path"], "error": repr(exc)})
        if index % 50 == 0 or index == len(rows):
            print(f"RENDER_STATE_BUILD={index}/{len(rows)}", flush=True)

    payload = {
        "strategy": "VST3/S2-partitioned-state-audio-verified-v1",
        "plugin_path": str(candidate.path),
        "class_id": template.class_id,
        "settings_source": "serum2_full_settings.settings_json",
        "state_filename_scheme": "{preset_id}.vstpreset",
        "state_dir": str(args.state_dir),
        "requested": len(rows),
        "built": len(reports),
        "failed": len(failures),
        "total_state_bytes": total_bytes,
        "mean_structural_coverage": (
            sum(row["structural_coverage"] for row in reports) / len(reports) if reports else 0.0
        ),
        "minimum_structural_coverage": (
            min(row["structural_coverage"] for row in reports) if reports else 0.0
        ),
        "unmatched_unique": len(unmatched),
        "unmatched_occurrences": sum(unmatched.values()),
        "unmatched_paths": dict(unmatched),
        "failures": failures,
        "presets": reports,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        "RENDER_STATE_RESULT="
        + json.dumps({key: payload[key] for key in (
            "requested",
            "built",
            "failed",
            "mean_structural_coverage",
            "minimum_structural_coverage",
            "unmatched_unique",
            "unmatched_occurrences",
            "total_state_bytes",
        )}, sort_keys=True)
    )
    return 0 if len(reports) == len(rows) == 710 and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

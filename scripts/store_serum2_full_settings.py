#!/usr/bin/env python3
"""Populate the authoritative full decoded Serum 2 settings store."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.db import DEFAULT_DB_PATH, Database
from core.serum2_preset import parse_serum2_preset


UPSERT_SQL = """
INSERT INTO serum2_full_settings(
  preset_id, metadata_json, settings_json, settings_sha256,
  payload_version, cbor_length, compressed_length
) VALUES (?,?,?,?,?,?,?)
ON CONFLICT(preset_id) DO UPDATE SET
  metadata_json=excluded.metadata_json,
  settings_json=excluded.settings_json,
  settings_sha256=excluded.settings_sha256,
  payload_version=excluded.payload_version,
  cbor_length=excluded.cbor_length,
  compressed_length=excluded.compressed_length,
  updated_at=CURRENT_TIMESTAMP
"""


def encode_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def snapshot(connection: sqlite3.Connection) -> dict[str, object]:
    statuses = connection.execute(
        "SELECT status, COUNT(*) FROM presets WHERE synth='serum2' GROUP BY status ORDER BY status"
    ).fetchall()
    return {
        "serum2_presets": int(
            connection.execute("SELECT COUNT(*) FROM presets WHERE synth='serum2'").fetchone()[0]
        ),
        "serum2_param_rows": int(
            connection.execute(
                "SELECT COUNT(*) FROM params WHERE preset_id IN "
                "(SELECT id FROM presets WHERE synth='serum2')"
            ).fetchone()[0]
        ),
        "serum2_statuses": {str(status): int(count) for status, count in statuses},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()
    database = Database(args.db)
    with database.connect() as connection:
        before = snapshot(connection)
        presets = connection.execute(
            "SELECT id,path FROM presets WHERE synth='serum2' ORDER BY id"
        ).fetchall()

    records = []
    failures = []
    total_json_bytes = 0
    for index, row in enumerate(presets, start=1):
        try:
            decoded = parse_serum2_preset(Path(row["path"]))
            metadata_json = encode_json(decoded.metadata)
            settings_json = encode_json(decoded.data)
            total_json_bytes += len(metadata_json.encode("utf-8")) + len(
                settings_json.encode("utf-8")
            )
            records.append(
                (
                    int(row["id"]),
                    metadata_json,
                    settings_json,
                    hashlib.sha256(settings_json.encode("utf-8")).hexdigest(),
                    int(decoded.payload_version),
                    int(decoded.cbor_length),
                    int(decoded.compressed_length),
                )
            )
        except Exception as exc:
            failures.append({"preset_id": int(row["id"]), "path": row["path"], "error": repr(exc)})
        if index % 50 == 0 or index == len(presets):
            print(f"FULL_SETTINGS_DECODE={index}/{len(presets)}", flush=True)

    if failures:
        print(json.dumps({"failures": failures}, indent=2))
        return 1

    with database.connect() as connection:
        connection.executemany(UPSERT_SQL, records)
        stored = int(connection.execute("SELECT COUNT(*) FROM serum2_full_settings").fetchone()[0])
        valid = int(
            connection.execute(
                "SELECT COUNT(*) FROM serum2_full_settings "
                "WHERE json_valid(metadata_json) AND json_valid(settings_json)"
            ).fetchone()[0]
        )
        joined = int(
            connection.execute(
                "SELECT COUNT(*) FROM serum2_full_settings f "
                "JOIN presets p ON p.id=f.preset_id WHERE p.synth='serum2'"
            ).fetchone()[0]
        )
        after = snapshot(connection)

    result = {
        "decoded": len(records),
        "stored": stored,
        "valid_json": valid,
        "joined_to_serum2": joined,
        "failures": len(failures),
        "json_bytes": total_json_bytes,
        "mapped_baseline_before": before,
        "mapped_baseline_after": after,
        "mapped_baseline_preserved": before == after,
        "authoritative_serum2_training_labels": "serum2_full_settings.settings_json",
    }
    print("FULL_SETTINGS_RESULT=" + json.dumps(result, sort_keys=True))
    return 0 if stored == len(presets) == valid == joined and before == after else 1


if __name__ == "__main__":
    raise SystemExit(main())

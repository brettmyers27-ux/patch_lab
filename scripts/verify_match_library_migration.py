#!/usr/bin/env python3
"""Prove the Match Library migration is additive against a copy of the real DB."""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.db import DEFAULT_DB_PATH, Database


LEGACY_TABLES = (
    "presets",
    "params",
    "renders",
    "fingerprints",
    "serum2_full_settings",
    "favorites",
)


def counts(path: Path) -> dict[str, int]:
    connection = sqlite3.connect(path)
    try:
        return {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in LEGACY_TABLES
        }
    finally:
        connection.close()


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="patchlab-migration-") as directory:
        copy = Path(directory) / "library.db"
        shutil.copy2(DEFAULT_DB_PATH, copy)
        connection = sqlite3.connect(copy)
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute("DROP TABLE IF EXISTS match_library")
        connection.execute("DROP TABLE IF EXISTS match_batches")
        connection.execute("DELETE FROM schema_migrations WHERE version=5")
        connection.commit()
        connection.close()
        before = counts(copy)
        Database(copy)
        after = counts(copy)
        connection = sqlite3.connect(copy)
        new_tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        connection.close()
    payload = {
        "source_database": str(DEFAULT_DB_PATH),
        "before": before,
        "after": after,
        "counts_unchanged": before == after,
        "new_tables_present": {
            "match_library",
            "match_batches",
        }.issubset(new_tables),
    }
    payload["gate_pass"] = payload["counts_unchanged"] and payload["new_tables_present"]
    print("MATCH_LIBRARY_MIGRATION=" + json.dumps(payload, sort_keys=True))
    return 0 if payload["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

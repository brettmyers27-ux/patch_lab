#!/usr/bin/env python3
"""Populate the path-based factory classification for the current catalog."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.db import DEFAULT_DB_PATH, Database
from core.platform_env import ENV


REPORT = PROJECT_ROOT / "data" / "models" / "factory_classification_report.json"


def main() -> int:
    database = Database(DEFAULT_DB_PATH)
    with database.connect() as connection:
        rows = connection.execute("SELECT id,path,synth FROM presets ORDER BY id").fetchall()
        updates = [
            (
                int(ENV.path_is_factory(str(row["synth"]), Path(str(row["path"])))),
                int(row["id"]),
            )
            for row in rows
        ]
        connection.executemany(
            "UPDATE presets SET is_factory=? WHERE id=?", updates
        )
        counts = connection.execute(
            "SELECT synth,is_factory,COUNT(*) AS count FROM presets "
            "GROUP BY synth,is_factory ORDER BY synth,is_factory DESC"
        ).fetchall()
    by_synth = {
        synth: {
            "factory": next(
                (
                    int(row["count"])
                    for row in counts
                    if row["synth"] == synth and int(row["is_factory"]) == 1
                ),
                0,
            ),
            "developer_owned": next(
                (
                    int(row["count"])
                    for row in counts
                    if row["synth"] == synth and int(row["is_factory"]) == 0
                ),
                0,
            ),
        }
        for synth in ("serum1", "serum2")
    }
    report = {
        "platform_branch": ENV.branch,
        "factory_roots": {
            synth: [str(path) for path in ENV.factory_roots_for(synth)]
            for synth in ("serum1", "serum2")
        },
        "by_synth": by_synth,
        "total_factory": sum(item["factory"] for item in by_synth.values()),
        "total_developer_owned": sum(
            item["developer_owned"] for item in by_synth.values()
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    for synth, item in by_synth.items():
        print(
            f"{synth}: factory={item['factory']:,}, "
            f"developer-owned={item['developer_owned']:,}"
        )
    print(f"FACTORY_CLASSIFICATION_REPORT={REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export presets for archived matches that never got auto-saved.

Batch 2 archived all 52 matches successfully but every export failed with
"Unknown Serum 2 base preset", fixed in core/serum2_preset_writer.py. The
matches were never re-run — this writes the presets that export should have
produced the first time, using the exact same verified worker every live
export goes through (scripts/export_match.py via app.worker_dispatch).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.db import Database
from core.match_batch import disambiguated_preset_path, sanitize_folder_name
from core.match_library import resolved_record_paths


def main() -> int:
    db_path = Path.home() / "Library/Application Support/Patch Lab/library.db"
    database = Database(db_path)
    library_root = Path.home() / "Library/Application Support/Patch Lab/match_library"

    with database.connect() as connection:
        rows = connection.execute(
            "SELECT match_uid,batch_id,source_name,recommendation_synth "
            "FROM match_library "
            "WHERE exported_preset_path IS NULL AND no_confident_match=0 "
            "ORDER BY id"
        ).fetchall()

    if not rows:
        print("Nothing to back-fill.")
        return 0

    print(f"{len(rows)} archived matches have no saved preset.")
    succeeded = 0
    failed: list[tuple[str, str]] = []

    for row in rows:
        uid = str(row["match_uid"])
        batch_id = row["batch_id"]
        synth = str(row["recommendation_synth"])
        extension = ".fxp" if synth == "serum1" else ".SerumPreset"
        record = database.get_match_library(uid)
        if record is None:
            continue
        _source, result_path = resolved_record_paths(record, library_root)

        if batch_id is not None:
            with database.connect() as connection:
                batch_row = connection.execute(
                    "SELECT export_folder FROM match_batches WHERE id=?", (batch_id,)
                ).fetchone()
            folder = Path(str(batch_row["export_folder"])) if batch_row else None
        else:
            folder = None
        if folder is None:
            token = "serum 2" if synth == "serum2" else "serum presets"
            base = "/Library/Audio/Presets/Xfer Records"
            folder = Path(
                f"{base}/Serum 2 Presets/Presets/User/PatchLab"
                if synth == "serum2"
                else f"{base}/Serum Presets/Presets/User/PatchLab"
            )
        folder.mkdir(parents=True, exist_ok=True)

        stem = sanitize_folder_name(Path(str(row["source_name"])).stem) or "Sound"
        output = disambiguated_preset_path(folder, f"PatchLab - {stem}", extension)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "app.worker_dispatch",
                "export",
                str(result_path),
                str(output),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        exported = None
        for line in result.stdout.splitlines():
            if line.startswith("EXPORT_RESULT="):
                exported = json.loads(line.removeprefix("EXPORT_RESULT="))
            elif line.startswith("EXPORT_ERROR="):
                failed.append((str(row["source_name"]), line.removeprefix("EXPORT_ERROR=")))
        if exported is None:
            if not failed or failed[-1][0] != str(row["source_name"]):
                failed.append((str(row["source_name"]), f"exit code {result.returncode}"))
            print(f"  FAIL  {row['source_name']}: {failed[-1][1][:80]}")
            continue
        database.set_match_exported_path(uid, Path(exported["path"]))
        succeeded += 1
        print(f"  OK    {row['source_name']} -> {Path(exported['path']).name}")

    print(f"\nSucceeded: {succeeded}/{len(rows)}")
    if failed:
        print(f"Failed: {len(failed)}")
        for name, error in failed:
            print(f"  {name}: {error[:120]}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

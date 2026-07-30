#!/usr/bin/env python3
"""Restart/reopen/lazy-preview/export gate for the latest durable match."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import soundfile as sf


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtWidgets import QApplication

from app.ui import MainWindow
from core.db import Database
from core.match_library import resolve_result_path, resolved_record_paths
from core.preview_cache import preview_cache_path, recommendation_cache_key


def main() -> int:
    application = QApplication.instance() or QApplication([])
    first = MainWindow()
    records = Database(first._match_database_path()).list_match_library()
    if not records:
        raise RuntimeError("Run one match before the persistence gate")
    record = records[0]
    source, result_path = resolved_record_paths(record, first._match_library_root())
    result = json.loads(result_path.read_text(encoding="utf-8"))
    source_audio, source_rate = sf.read(source, dtype="float32")
    first.close()
    application.processEvents()

    restarted = MainWindow()
    restarted.open_library_match(record.match_uid)
    application.processEvents()
    reopened_rows = restarted.existing_list_layout.count() - 1
    recommendation = result.get("recommendation")
    preview_ok = export_ok = False
    export_payload: dict = {}
    if isinstance(recommendation, dict):
        cache_key = recommendation_cache_key(result_path, recommendation)
        cache_root = restarted._preview_cache_root()
        if (
            not recommendation.get("meaningfully_modified", False)
            and recommendation.get("preview_source_path")
        ):
            preview_command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "render_factory_preview.py"),
                str(recommendation["preview_source_path"]),
                "--synth",
                str(recommendation["synth"]),
                "--note",
                "60",
                "--content-hash",
                cache_key,
                "--output-root",
                str(cache_root),
            ]
        else:
            preview_command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "render_recommendation_preview.py"),
                str(result_path),
                "--note",
                "60",
                "--cache-key",
                cache_key,
                "--output-root",
                str(cache_root),
            ]
        preview = subprocess.run(
            preview_command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        preview_ok = (
            preview.returncode == 0
            and preview_cache_path(cache_root, cache_key, 60).is_file()
        )
        with tempfile.TemporaryDirectory(prefix="patchlab-library-export-") as directory:
            extension = ".fxp" if record.recommendation_synth == "serum1" else ".SerumPreset"
            output = Path(directory) / f"Verified{extension}"
            exported = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "export_match.py"),
                    str(result_path),
                    str(output),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            export_ok = exported.returncode == 0 and output.is_file()
            line = next(
                (item for item in exported.stdout.splitlines() if item.startswith("EXPORT_RESULT=")),
                None,
            )
            export_payload = json.loads(line.split("=", 1)[1]) if line else {
                "stdout": exported.stdout,
                "stderr": exported.stderr,
            }
    else:
        preview_ok = export_ok = record.no_confident_match
    stored_source = result.get("source", {}).get("path", "")
    candidate_ok = True
    winner_ok = True
    if isinstance(recommendation, dict):
        candidate_ok = resolve_result_path(
            result_path, recommendation["candidate_path"]
        ).is_file()
        winner_ok = resolve_result_path(
            result_path, recommendation["winner_audio_path"]
        ).is_file()
    payload = {
        "match_uid": record.match_uid,
        "database_paths_relative": (
            not record.source_audio_path.is_absolute()
            and not record.result_json_path.is_absolute()
        ),
        "stored_source_path_relative": bool(stored_source) and not Path(stored_source).is_absolute(),
        "source_decodable": source_audio.size > 0 and source_rate > 0,
        "candidate_present": candidate_ok,
        "winner_present": winner_ok,
        "restart_reopen_existing_rows": reopened_rows,
        "octave_preview_pass": preview_ok,
        "verified_export_pass": export_ok,
        "export": export_payload,
    }
    payload["gate_pass"] = all(
        (
            payload["database_paths_relative"],
            payload["stored_source_path_relative"],
            payload["source_decodable"],
            payload["candidate_present"],
            payload["winner_present"],
            payload["restart_reopen_existing_rows"] == 10,
            payload["octave_preview_pass"],
            payload["verified_export_pass"],
        )
    )
    print("MATCH_LIBRARY_PERSISTENCE=" + json.dumps(payload, sort_keys=True))
    restarted.close()
    return 0 if payload["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

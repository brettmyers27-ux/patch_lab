#!/usr/bin/env python3
"""Exercise the real Match UI QProcess path for raw, MP3, and silence gates."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = Path(
    os.environ.get(
        "PATCHLAB_GATE_PROJECT_ROOT",
        str(Path(__file__).resolve().parents[1]),
    )
).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from app.ui import MainWindow
from app.workers import MatchProcessRunner


REPORT = PROJECT_ROOT / "data" / "models" / "milestone4_ui_gate_report.json"
FIXTURE_ID = 67


def _encode_mp3(source: Path, output: Path) -> None:
    import imageio_ffmpeg

    completed = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-i",
            str(source),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "96k",
            "-y",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr)


def _run_ui_worker(path: Path) -> tuple[Path, list[dict[str, Any]]]:
    runner = MatchProcessRunner()
    loop = QEventLoop()
    result: list[str] = []
    errors: list[str] = []
    progress: list[dict[str, Any]] = []
    runner.completed.connect(lambda value: (result.append(value), loop.quit()))
    runner.failed.connect(lambda value: (errors.append(value), loop.quit()))
    runner.progress.connect(progress.append)
    timeout = QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(lambda: (errors.append("UI match worker timed out"), runner.cancel(), loop.quit()))
    timeout.start(600_000)
    mapping = os.environ.get("PATCHLAB_GATE_FACTORY_MAPPING")
    runner.start(
        path,
        target_synth="serum2",
        budget="quick",
        offset=0.0,
        factory_only=bool(mapping),
        factory_mapping=Path(mapping).expanduser().resolve() if mapping else None,
    )
    loop.exec()
    timeout.stop()
    if errors:
        raise RuntimeError(errors[0])
    if not result:
        raise RuntimeError("UI match worker produced no result")
    return Path(result[0]), progress


def main() -> int:
    application = QApplication.instance() or QApplication([])
    raw = PROJECT_ROOT / "data" / "audio" / str(FIXTURE_ID) / "60.wav"
    if not raw.is_file():
        raise FileNotFoundError(raw)
    with tempfile.TemporaryDirectory(prefix="patchlab-ui-gate-") as temporary:
        directory = Path(temporary)
        mp3 = directory / "lossy.mp3"
        silence = directory / "silence.wav"
        _encode_mp3(raw, mp3)
        sf.write(silence, np.zeros(5 * 44_100, dtype=np.float32), 44_100)

        raw_path, raw_progress = _run_ui_worker(raw)
        mp3_path, mp3_progress = _run_ui_worker(mp3)
        silence_path, silence_progress = _run_ui_worker(silence)
        raw_result = json.loads(raw_path.read_text(encoding="utf-8"))
        mp3_result = json.loads(mp3_path.read_text(encoding="utf-8"))
        silence_result = json.loads(silence_path.read_text(encoding="utf-8"))
        with sqlite3.connect(PROJECT_ROOT / "data" / "library.db") as connection:
            fixture_hash = str(
                connection.execute(
                    "SELECT content_hash FROM presets WHERE id=?", (FIXTURE_ID,)
                ).fetchone()[0]
            )

        raw_rank = next(
            (
                index + 1
                for index, item in enumerate(raw_result["existing_matches"])
                if str(item.get("content_hash")) == fixture_hash
            ),
            None,
        )
        mp3_rank = next(
            (
                index + 1
                for index, item in enumerate(mp3_result["existing_matches"])
                if str(item.get("content_hash")) == fixture_hash
            ),
            None,
        )

        window = MainWindow()
        window._show_match_result(raw_result)
        raw_ui_rows = window.existing_list_layout.count() - 1  # minus trailing stretch
        raw_ui_score = window.recommendation_confidence.text()
        window._show_match_result(silence_result)
        silence_ui_text = window.recommendation_confidence.text()
        default_target = str(window.match_synth.currentData())
        budget_count = window.match_budget.count()
        window.close()

    report = {
        "fixture_preset_id": FIXTURE_ID,
        "fixture_content_hash": fixture_hash,
        "raw": {
            "decoder": raw_result["source"]["decoder"],
            "own_preset_rank": raw_rank,
            "top_preset_id": raw_result["existing_matches"][0]["preset_id"],
            "ui_rows": raw_ui_rows,
            "ui_confidence_text": raw_ui_score,
            "progress_updates": len(raw_progress),
            "pass": raw_rank == 1 and raw_ui_rows == 10 and bool(raw_progress),
        },
        "lossy_mp3": {
            "decoder": mp3_result["source"]["decoder"],
            "own_preset_rank": mp3_rank,
            "progress_updates": len(mp3_progress),
            "pass": mp3_rank is not None and mp3_rank <= 5 and bool(mp3_progress),
        },
        "silence": {
            "decoder": silence_result["source"]["decoder"],
            "no_confident_match": silence_result["no_confident_match"],
            "recommendation_is_none": silence_result["recommendation"] is None,
            "ui_message": silence_ui_text,
            "progress_updates": len(silence_progress),
            "pass": (
                silence_result["no_confident_match"]
                and silence_result["recommendation"] is None
                and "No confident match" in silence_ui_text
            ),
        },
        "controls": {
            "default_target_synth": default_target,
            "search_budget_count": budget_count,
            "pass": default_target == "serum2" and budget_count == 3,
        },
    }
    report["gate_pass"] = all(
        report[name]["pass"] for name in ("raw", "lossy_mp3", "silence", "controls")
    )
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    for name in ("raw", "lossy_mp3", "silence", "controls"):
        print(
            f"{'PASS' if report[name]['pass'] else 'FAIL'} {name}: "
            + json.dumps(report[name], sort_keys=True),
            flush=True,
        )
    print(f"MILESTONE4_UI_GATE_PASS={str(report['gate_pass']).lower()}", flush=True)
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

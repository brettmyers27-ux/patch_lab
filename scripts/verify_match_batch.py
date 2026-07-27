#!/usr/bin/env python3
"""Real sequential batch, cancel/resume, no-overwrite, and no-match gate."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from app.ui import MainWindow
from core.db import Database
from core.match_batch import discover_batch_audio, resumable_batch_files


FIXTURES = (67, 121, 412, 993)


def wait_for_batch(window: MainWindow, timeout_ms: int = 1_200_000) -> None:
    loop = QEventLoop()
    poll = QTimer()
    poll.timeout.connect(lambda: loop.quit() if window._batch_state is None else None)
    poll.start(100)
    timeout = QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(loop.quit)
    timeout.start(timeout_ms)
    loop.exec()
    poll.stop()
    if window._batch_state is not None:
        raise RuntimeError("Batch gate timed out")


def configure(
    window: MainWindow,
    *,
    batch_id: int,
    source: Path,
    output: Path,
    supported: tuple[Path, ...],
    unsupported: int,
    cancel: bool,
) -> tuple[int, int]:
    database = Database(window._match_database_path())
    completed_hashes = database.batch_completed_hashes(batch_id)
    pending, resumed = resumable_batch_files(list(supported), completed_hashes)
    existing = database.get_match_batch(batch_id)
    window._batch_state = {
        "batch_id": batch_id,
        "folder_name": "Gate Batch",
        "source_folder": source,
        "export_folder": output,
        "target_synth": "serum2",
        "budget": "quick",
        "files": pending,
        "index": 0,
        "total": len(supported),
        "completed": len(completed_hashes),
        "failed": existing.failed_files if existing else 0,
        "skipped": resumed + unsupported,
        "unsupported": unsupported,
        "started": time.monotonic(),
        "cancel_requested": False,
        "phase": "idle",
    }
    database.update_match_batch(
        batch_id,
        total_files=len(supported),
        completed_files=len(completed_hashes),
        failed_files=existing.failed_files if existing else 0,
        status="running",
    )
    window.batch_button.setEnabled(False)
    window.library_batch_cancel.setEnabled(True)
    window.append_log(
        f"Gate batch started: {len(pending)} pending, {resumed} resume-skipped, {unsupported} unsupported-skipped"
    )
    window._start_next_batch_file()
    if cancel:
        QTimer.singleShot(100, window.cancel_match_batch)
    return len(pending), resumed


def main() -> int:
    application = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory(prefix="patchlab-batch-gate-") as directory:
        root = Path(directory)
        source = root / "source"
        output = root / "PatchLab" / "Gate Batch"
        source.mkdir()
        output.mkdir(parents=True)
        for position, preset_id in enumerate(FIXTURES):
            shutil.copy2(
                PROJECT_ROOT / "data" / "audio" / str(preset_id) / "60.wav",
                source / f"{position:02d}-sound-{preset_id}.wav",
            )
        sf.write(
            source / "99-silence.wav",
            np.zeros(48_000, dtype=np.float32),
            48_000,
        )
        (source / "unsupported.txt").write_text("must be reported")
        sentinel = output / "PatchLab - 00-sound-67.SerumPreset"
        sentinel.write_bytes(b"DO NOT OVERWRITE")
        discovery = discover_batch_audio(source)

        window = MainWindow()
        window.match_synth.setCurrentIndex(0)
        window.match_budget.setCurrentIndex(0)
        database = Database(window._match_database_path())
        batch_id = database.create_match_batch(
            folder_name="Gate Batch",
            source_folder=source,
            export_folder=output,
            target_synth="serum2",
            budget="quick",
            total_files=len(discovery.supported),
        )
        configure(
            window,
            batch_id=batch_id,
            source=source,
            output=output,
            supported=discovery.supported,
            unsupported=discovery.unsupported_count,
            cancel=True,
        )
        wait_for_batch(window)
        cancelled = database.get_match_batch(batch_id)
        completed_before_resume = cancelled.completed_files if cancelled else 0

        pending_after_cancel, resumed = configure(
            window,
            batch_id=batch_id,
            source=source,
            output=output,
            supported=discovery.supported,
            unsupported=discovery.unsupported_count,
            cancel=False,
        )
        wait_for_batch(window)
        final = database.get_match_batch(batch_id)
        records = [
            record
            for record in database.list_match_library()
            if record.batch_id == batch_id
        ]
        exported = [record for record in records if record.exported_preset_path]
        no_match = [record for record in records if record.no_confident_match]
        exported_exist = all(
            record.exported_preset_path and record.exported_preset_path.is_file()
            for record in exported
        )
        preset_files = list(output.glob("*.SerumPreset"))
        sentinel_safe = sentinel.read_bytes() == b"DO NOT OVERWRITE"
        window.nav_tabs.setCurrentIndex(1)
        application.processEvents()
        log = window.log_pane.toPlainText()
        payload = {
            "supported": len(discovery.supported),
            "unsupported_skipped": discovery.unsupported_count,
            "cancelled_status": cancelled.status if cancelled else None,
            "completed_before_resume": completed_before_resume,
            "resume_hashes_skipped": resumed,
            "pending_after_cancel": pending_after_cancel,
            "final_status": final.status if final else None,
            "final_completed": final.completed_files if final else None,
            "final_failed": final.failed_files if final else None,
            "library_entries": len(records),
            "no_confident_entries": len(no_match),
            "verified_exports": len(exported),
            "exported_files_exist": exported_exist,
            "preset_files_including_sentinel": len(preset_files),
            "preexisting_file_unchanged": sentinel_safe,
            "unsupported_reported_in_log": "unsupported-skipped" in log,
        }
        payload["gate_pass"] = all(
            (
                payload["unsupported_skipped"] == 1,
                payload["cancelled_status"] == "cancelled",
                payload["completed_before_resume"] >= 1,
                payload["resume_hashes_skipped"] >= 1,
                payload["final_status"] == "complete",
                payload["final_completed"] == len(discovery.supported),
                payload["library_entries"] == len(discovery.supported),
                payload["no_confident_entries"] >= 1,
                payload["verified_exports"] + payload["no_confident_entries"]
                == len(discovery.supported),
                payload["exported_files_exist"],
                payload["preexisting_file_unchanged"],
                payload["unsupported_reported_in_log"],
            )
        )
        print("MATCH_BATCH_GATE=" + json.dumps(payload, sort_keys=True))
        window.close()
        return 0 if payload["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

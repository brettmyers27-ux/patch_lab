#!/usr/bin/env python3
"""Exercise real QProcess workers from inside the frozen PatchLab executable."""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf
from PySide6.QtCore import QCoreApplication, QEventLoop, QProcess, QTimer

from app.workers import (
    AnalyzeProcessRunner,
    ExportProcessRunner,
    MatchProcessRunner,
    PreviewProcessRunner,
    RenderProcessRunner,
    ScanProcessRunner,
)


def _child_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(usage.ru_utime + usage.ru_stime)


def _run(
    runner: Any,
    start: Callable[[], None],
    *,
    timeout_ms: int,
) -> dict[str, Any]:
    loop = QEventLoop()
    completed: list[Any] = []
    failures: list[str] = []
    logs: list[str] = []
    progress: list[Any] = []
    child_pids: list[int] = []
    runner.completed.connect(lambda value: (completed.append(value), loop.quit()))
    runner.failed.connect(lambda message: (failures.append(message), loop.quit()))
    runner.log.connect(logs.append)
    if hasattr(runner, "progress"):
        runner.progress.connect(lambda *value: progress.append(value))
    runner.process.started.connect(
        lambda: child_pids.append(int(runner.process.processId()))
    )
    guard = QTimer()
    guard.setSingleShot(True)
    timed_out: list[bool] = []

    def timeout() -> None:
        timed_out.append(True)
        if runner.process.state() != QProcess.ProcessState.NotRunning:
            runner.process.kill()
        loop.quit()

    guard.timeout.connect(timeout)
    cpu_before = _child_cpu_seconds()
    started = time.monotonic()
    start()
    guard.start(timeout_ms)
    loop.exec()
    guard.stop()
    runner.process.waitForFinished(5_000)
    elapsed = time.monotonic() - started
    cpu_seconds = max(0.0, _child_cpu_seconds() - cpu_before)
    return {
        "completed": completed[0] if completed else None,
        "failure": failures[0] if failures else None,
        "timed_out": bool(timed_out),
        "elapsed_s": elapsed,
        "child_cpu_s": cpu_seconds,
        "child_pid": child_pids[0] if child_pids else None,
        "progress_updates": len(progress),
        "ready_logged": any(line.startswith("Worker ready:") for line in logs),
        "log_tail": logs[-8:],
    }


def _prepare_render_database(source: Path, output: Path, preset_id: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    with sqlite3.connect(output) as connection:
        connection.execute("DELETE FROM renders")
        connection.execute(
            "UPDATE presets SET status='params_dumped',error=NULL WHERE id=?",
            (preset_id,),
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=Path, action="append", required=True)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--factory-mapping", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--render-preset-id", type=int, default=1)
    parser.add_argument("--play-preview", action="store_true")
    args = parser.parse_args()
    if len(args.query) < 5:
        parser.error("at least five --query files are required")

    application = QCoreApplication.instance() or QCoreApplication([])
    work_root = args.work_root.expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    render_db = work_root / "render" / "library.db"
    render_audio = work_root / "render" / "audio"
    _prepare_render_database(
        args.source_db.expanduser().resolve(),
        render_db,
        args.render_preset_id,
    )

    report: dict[str, Any] = {
        "frozen": bool(getattr(__import__("sys"), "frozen", False)),
        "executable": __import__("sys").executable,
        "matches": [],
    }
    result_paths: list[Path] = []
    for index, query in enumerate(args.query[:5], 1):
        runner = MatchProcessRunner()
        outcome = _run(
            runner,
            lambda runner=runner, query=query: runner.start(
                query.expanduser().resolve(),
                target_synth="serum2",
                budget="quick",
                offset=0.0,
                session_root=work_root / "matches",
                factory_only=True,
                factory_mapping=args.factory_mapping.expanduser().resolve(),
            ),
            timeout_ms=180_000,
        )
        outcome["index"] = index
        outcome["query"] = str(query)
        if outcome["completed"]:
            result_path = Path(str(outcome["completed"]))
            result_paths.append(result_path)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            outcome["top_similarity"] = float(
                result["existing_matches"][0]["similarity"]
            )
            outcome["recommendation_origin"] = result["recommendation"]["origin"]
        report["matches"].append(outcome)

    first_result = result_paths[0] if result_paths else None
    if first_result is not None:
        exported = work_root / "export" / "PatchLab Packaged Gate.SerumPreset"
        export_runner = ExportProcessRunner()
        report["export"] = _run(
            export_runner,
            lambda: export_runner.start(first_result, exported),
            timeout_ms=120_000,
        )
        report["export"]["file_exists"] = exported.is_file()

        result = json.loads(first_result.read_text(encoding="utf-8"))
        recommendation = result["recommendation"]
        preview_runner = PreviewProcessRunner()
        preview = _run(
            preview_runner,
            lambda: preview_runner.start(
                Path(recommendation["factory_source_path"]),
                synth=str(recommendation["synth"]),
                midi_note=60,
                content_hash=str(recommendation["content_hash"]),
                output_root=work_root / "preview",
            ),
            timeout_ms=180_000,
        )
        preview_path = (
            Path(str(preview["completed"])) if preview["completed"] else None
        )
        preview["file_exists"] = bool(preview_path and preview_path.is_file())
        preview["rms_dbfs"] = None
        preview["playback_succeeded"] = not args.play_preview
        if preview_path and preview_path.is_file():
            audio, sample_rate = sf.read(
                preview_path, dtype="float32", always_2d=True
            )
            rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
            preview["rms_dbfs"] = 20.0 * np.log10(max(rms, 1e-12))
            if args.play_preview:
                try:
                    import sounddevice as sd

                    sd.play(audio[:sample_rate], sample_rate, blocking=True)
                    preview["playback_succeeded"] = True
                except Exception as exc:
                    preview["playback_error"] = f"{type(exc).__name__}: {exc}"
        report["preview"] = preview

    render_runner = RenderProcessRunner()
    report["render"] = _run(
        render_runner,
        lambda: render_runner.start(
            db_path=render_db,
            audio_root=render_audio,
            state_dir=args.state_dir.expanduser().resolve(),
            preset_ids=[args.render_preset_id],
            workers=1,
        ),
        timeout_ms=300_000,
    )
    with sqlite3.connect(render_db) as connection:
        report["render"]["persisted_rows"] = int(
            connection.execute(
                "SELECT COUNT(*) FROM renders WHERE preset_id=?",
                (args.render_preset_id,),
            ).fetchone()[0]
        )

    analyze_runner = AnalyzeProcessRunner()
    report["analyze"] = _run(
        analyze_runner,
        lambda: analyze_runner.start(
            False,
            smoke_db=render_db,
            smoke_feature_dir=work_root / "analyze-features",
        ),
        timeout_ms=300_000,
    )

    empty_scan = work_root / "empty-scan"
    empty_scan.mkdir(exist_ok=True)
    old_timeout = os.environ.get("PATCHLAB_WORKER_STARTUP_TIMEOUT_MS")
    old_delay = os.environ.get("PATCHLAB_WORKER_TEST_READY_DELAY_SECONDS")
    os.environ["PATCHLAB_WORKER_STARTUP_TIMEOUT_MS"] = "300"
    os.environ["PATCHLAB_WORKER_TEST_READY_DELAY_SECONDS"] = "3"
    try:
        broken_runner = ScanProcessRunner()
        report["fail_fast"] = _run(
            broken_runner,
            lambda: broken_runner.start(empty_scan, local_library=True),
            timeout_ms=5_000,
        )
    finally:
        if old_timeout is None:
            os.environ.pop("PATCHLAB_WORKER_STARTUP_TIMEOUT_MS", None)
        else:
            os.environ["PATCHLAB_WORKER_STARTUP_TIMEOUT_MS"] = old_timeout
        if old_delay is None:
            os.environ.pop("PATCHLAB_WORKER_TEST_READY_DELAY_SECONDS", None)
        else:
            os.environ["PATCHLAB_WORKER_TEST_READY_DELAY_SECONDS"] = old_delay

    matches_pass = (
        len(report["matches"]) == 5
        and all(
            item["completed"]
            and item["ready_logged"]
            and item["child_cpu_s"] > 0
            and item["progress_updates"] > 0
            for item in report["matches"]
        )
    )
    report["gate_pass"] = bool(
        report["frozen"]
        and matches_pass
        and report.get("export", {}).get("file_exists")
        and report.get("preview", {}).get("file_exists")
        and report.get("preview", {}).get("rms_dbfs", -120.0) > -60.0
        and report.get("preview", {}).get("playback_succeeded")
        and report["render"]["completed"]
        and report["render"]["persisted_rows"] == 7
        and report["render"]["child_cpu_s"] > 0
        and report["analyze"]["completed"]
        and report["analyze"]["child_cpu_s"] > 0
        and report["fail_fast"]["failure"]
        and "did not confirm startup" in report["fail_fast"]["failure"]
        and report["fail_fast"]["elapsed_s"] < 2.0
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        "PACKAGED_WORKER_GATE=" + json.dumps(report, sort_keys=True),
        flush=True,
    )
    del application
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import Mock

from PySide6.QtCore import QCoreApplication, QEventLoop, QProcess, QTimer

import app.workers as workers
from app.workers import (
    AnalyzeProcessRunner,
    ExportProcessRunner,
    MatchProcessRunner,
    PreviewProcessRunner,
    RenderProcessRunner,
    ScanProcessRunner,
)
from core.worker_runtime import WORKER_FLAG, worker_invocation


def test_dispatcher_emits_ready_before_worker_usage() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "app.worker_dispatch", "match", "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 0
    assert completed.stdout.splitlines()[0] == "PATCHLAB_WORKER_READY=match"


def test_frozen_and_development_invocations(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delattr(sys, "frozen", raising=False)
    program, arguments = worker_invocation("match", ["fixture.wav"])
    assert program == sys.executable
    assert arguments == [
        "-m",
        "app.worker_dispatch",
        "match",
        "fixture.wav",
    ]

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    program, arguments = worker_invocation("match", ["fixture.wav"])
    assert program == sys.executable
    assert arguments == [WORKER_FLAG, "match", "fixture.wav"]


def test_windows_pythonw_parent_uses_console_python_for_workers(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    pythonw = tmp_path / "pythonw.exe"
    python = tmp_path / "python.exe"
    pythonw.touch()
    python.touch()
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "executable", str(pythonw))

    program, arguments = worker_invocation("match", ["fixture.wav"])

    assert program == str(python)
    assert arguments == [
        "-m",
        "app.worker_dispatch",
        "match",
        "fixture.wav",
    ]


def test_windows_crlf_worker_handshake_is_accepted() -> None:
    runner = MatchProcessRunner()
    runner._worker_name = "match"

    handled = runner._handle_worker_line("PATCHLAB_WORKER_READY=match\r")

    assert handled
    assert runner._worker_ready
    assert not runner._startup_failure_emitted


def test_every_qprocess_runner_uses_shared_dispatch() -> None:
    cases: list[tuple[object, str, tuple, dict]] = [
        (ScanProcessRunner(), "scan", (Path("/tmp/presets"),), {}),
        (
            ScanProcessRunner(),
            "local-library",
            (Path("/tmp/presets"),),
            {"local_library": True},
        ),
        (RenderProcessRunner(), "render-library", (), {}),
        (AnalyzeProcessRunner(), "analyze", (False,), {}),
        (
            MatchProcessRunner(),
            "match",
            (Path("/tmp/query.wav"),),
            {
                "target_synth": "serum2",
                "budget": "quick",
                "offset": 0.0,
            },
        ),
        (
            ExportProcessRunner(),
            "export",
            (Path("/tmp/result.json"), Path("/tmp/output.SerumPreset")),
            {},
        ),
        (
            PreviewProcessRunner(),
            "factory-preview",
            (Path("/tmp/source.SerumPreset"),),
            {
                "synth": "serum2",
                "midi_note": 60,
                "content_hash": "abc",
            },
        ),
    ]
    for runner, expected, positional, keywords in cases:
        start_worker = Mock()
        runner._start_worker = start_worker  # type: ignore[attr-defined]
        runner.start(*positional, **keywords)  # type: ignore[attr-defined]
        assert start_worker.call_args.args[0] == expected

    recommendation = PreviewProcessRunner()
    recommendation._start_worker = Mock()  # type: ignore[method-assign]
    recommendation.start_recommendation(Path("/tmp/result.json"), 60)
    assert recommendation._start_worker.call_args.args[0] == (  # type: ignore[attr-defined]
        "recommendation-preview"
    )


def test_missing_handshake_fails_fast_and_kills_process(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    application = QCoreApplication.instance() or QCoreApplication([])
    monkeypatch.setenv("PATCHLAB_WORKER_STARTUP_TIMEOUT_MS", "100")
    monkeypatch.setattr(
        workers,
        "worker_invocation",
        lambda _name, _arguments: (
            sys.executable,
            ["-c", "import time; time.sleep(10)"],
        ),
    )
    runner = ScanProcessRunner()
    failures: list[str] = []
    loop = QEventLoop()
    runner.failed.connect(lambda message: (failures.append(message), loop.quit()))
    guard = QTimer()
    guard.setSingleShot(True)
    guard.timeout.connect(loop.quit)
    started = time.monotonic()
    runner.start(Path("/tmp/presets"))
    guard.start(2_000)
    loop.exec()
    elapsed = time.monotonic() - started
    runner.process.waitForFinished(2_000)
    application.processEvents()

    assert elapsed < 1.0
    assert failures
    assert "did not confirm startup" in failures[0]
    assert "stopped instead of being left hung" in failures[0]
    assert runner.process.state() == QProcess.ProcessState.NotRunning

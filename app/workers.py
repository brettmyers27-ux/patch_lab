"""QProcess-backed background jobs and worker-process entry points."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from core.worker_runtime import (
    DEFAULT_STARTUP_TIMEOUT_MS,
    WORKER_READY_PREFIX,
    worker_invocation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTHENTICATED_WORKERS = frozenset({"local-library", "bug-report", "check-update", "download-update"})

class _ProcessRunnerBase(QObject):
    """Shared QProcess launch and bounded worker-startup handshake."""

    def _init_worker_process(self) -> None:
        self.process = QProcess(self)
        self.process.setProcessChannelMode(
            QProcess.ProcessChannelMode.MergedChannels
        )
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._process_error)
        self._startup_timer = QTimer(self)
        self._startup_timer.setSingleShot(True)
        self._startup_timer.timeout.connect(self._startup_timed_out)
        self._worker_name = ""
        self.operation_id = ""
        self._worker_ready = False
        self._startup_failure_emitted = False
        self._buffer = ""

    def _pop_line(self) -> str | None:
        """Return the next complete worker output line, or None if none is buffered.

        Windows' text-mode stdout translates every printed '\\n' to '\\r\\n', so
        a worker running there emits handshake, progress, and result lines all
        carrying a trailing '\\r'. Splitting on bare '\\n' alone leaves that '\\r'
        attached to `line`, which silently breaks exact-string comparisons —
        this is what previously made every worker (scan, render, analyze,
        match, export, preview) report "wrong startup handshake" on Windows
        while working correctly on macOS. `rstrip("\\r")` is a no-op on
        properly-terminated Unix output, so this is safe on every platform.
        """

        if "\n" not in self._buffer:
            return None
        line, self._buffer = self._buffer.split("\n", 1)
        return line.rstrip("\r")

    def _start_worker(self, worker_name: str, arguments: list[str]) -> None:
        self._worker_name = worker_name
        self._worker_ready = False
        self._startup_failure_emitted = False
        program, invocation = worker_invocation(worker_name, arguments)
        # One correlation ID per worker launch, handed to the child through its
        # environment, so a GUI-side event and the worker's own events for the
        # same operation can be joined in a support bundle. Failure here must
        # never stop a worker from starting.
        self.operation_id = ""
        try:
            from core.diagnostics import child_environment, new_operation_id

            self.operation_id = new_operation_id(worker_name.replace("_", "-"))
            environment = QProcessEnvironment.systemEnvironment()
            for key, value in child_environment(self.operation_id).items():
                environment.insert(key, value)
            # The GUI alone owns the durable device session. Workers receive
            # only a one-hour access token and cannot read the session file.
            from core.access_gate import AccessManager, WORKER_ACCESS_TOKEN_ENV

            token = AccessManager().access_token() if worker_name in AUTHENTICATED_WORKERS else None
            if token:
                environment.insert(WORKER_ACCESS_TOKEN_ENV, token)
            else:
                environment.remove(WORKER_ACCESS_TOKEN_ENV)
            self.process.setProcessEnvironment(environment)
        except Exception:
            self.operation_id = ""
        try:
            timeout_ms = int(
                os.environ.get(
                    "PATCHLAB_WORKER_STARTUP_TIMEOUT_MS",
                    str(DEFAULT_STARTUP_TIMEOUT_MS),
                )
            )
        except ValueError:
            timeout_ms = DEFAULT_STARTUP_TIMEOUT_MS
        self._startup_timer.start(max(50, timeout_ms))
        self.process.start(program, invocation)

    def _handle_worker_line(self, line: str) -> bool:
        if not line.startswith(WORKER_READY_PREFIX):
            return False
        actual = line.removeprefix(WORKER_READY_PREFIX)
        if self._worker_ready:
            # Analyze & Learn intentionally launches child phases through the
            # same dispatcher. Their sentinels are evidence of nested startup,
            # not a replacement for the already-validated outer handshake.
            self.log.emit(f"Nested worker ready: {actual}")
            return True
        if actual != self._worker_name:
            self._emit_startup_failure(
                f"{self._worker_name} worker returned the wrong startup handshake "
                f"({actual or 'empty'})"
            )
            self.process.kill()
            return True
        self._worker_ready = True
        self._startup_timer.stop()
        self.log.emit(f"Worker ready: {actual}")
        return True

    def _startup_timed_out(self) -> None:
        if self._worker_ready or self._startup_failure_emitted:
            return
        timeout_seconds = max(
            0.05,
            self._startup_timer.interval() / 1000.0,
        )
        self._emit_startup_failure(
            f"{self._worker_name} worker did not confirm startup within "
            f"{timeout_seconds:g}s; it was stopped instead of being left hung"
        )
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()

    def _process_error(self, error: QProcess.ProcessError) -> None:
        if self._worker_ready or self._startup_failure_emitted:
            return
        self._startup_timer.stop()
        self._emit_startup_failure(
            f"{self._worker_name} worker could not start: "
            f"{self.process.errorString()} ({error.name})"
        )

    def _emit_startup_failure(self, message: str) -> None:
        if self._startup_failure_emitted:
            return
        self._startup_failure_emitted = True
        self.failed.emit(message)

    def _finished_before_ready(self, exit_code: int) -> bool:
        self._startup_timer.stop()
        if self._startup_failure_emitted:
            return True
        if not self._worker_ready:
            self._emit_startup_failure(
                f"{self._worker_name} worker exited with code {exit_code} "
                "before its startup handshake"
            )
            return True
        return False


class ScanProcessRunner(_ProcessRunnerBase):
    log = Signal(str)
    progress = Signal(int, int)
    stage_progress = Signal(dict)
    completed = Signal(dict)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._init_worker_process()
        self._buffer = ""
        self._summary: dict[str, int] | None = None

    @property
    def running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def start(
        self,
        root: Path | None = None,
        *,
        local_library: bool = False,
        fingerprint_only: bool = False,
        refresh_only: bool = False,
        preset_ids: list[int] | None = None,
        workers: int = 4,
        pending_generation: str | None = None,
    ) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            raise RuntimeError("Scan worker is already running")
        self._buffer = ""
        self._summary = None
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        if pending_generation is not None:
            # Process presets already catalogued as pending for one generation.
            # No folder argument: rediscovering the library is exactly what this
            # path exists to avoid.
            if pending_generation not in {"serum1", "serum2"}:
                raise ValueError(f"Unknown generation {pending_generation!r}")
            self._start_worker(
                "process-pending",
                ["--generation", pending_generation, "--workers", str(max(1, workers))],
            )
        elif refresh_only:
            assert root is not None
            self._start_worker("refresh-library", [str(root)])
        elif fingerprint_only:
            # No folder to scan here: this fingerprints whatever is already
            # rendered but missing from the fingerprints table, regardless of
            # which pipeline rendered it. Same LOCAL_LIBRARY_* output protocol
            # as the local-library worker, so this class's own parsing below
            # already understands it without any changes.
            self._start_worker("fingerprint-local", [])
        elif local_library:
            assert root is not None
            if workers < 1:
                raise ValueError("workers must be positive")
            arguments = [str(root), "--workers", str(workers)]
            for preset_id in preset_ids or ():
                arguments.extend(["--preset-id", str(preset_id)])
            self._start_worker("local-library", arguments)
        else:
            assert root is not None
            self._start_worker("scan", ["--scan", str(root)])

    def cancel(self) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.terminate()

    def _read_output(self) -> None:
        self._buffer += bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        while (line := self._pop_line()) is not None:
            if self._handle_worker_line(line):
                continue
            if line.startswith("WORKER_PROGRESS="):
                current, total = line.removeprefix("WORKER_PROGRESS=").split("/", 1)
                self.progress.emit(int(current), int(total))
            elif line.startswith("LOCAL_LIBRARY_PROGRESS="):
                self.stage_progress.emit(
                    json.loads(line.removeprefix("LOCAL_LIBRARY_PROGRESS="))
                )
            elif line.startswith("SCAN_SUMMARY="):
                self._summary = json.loads(line.removeprefix("SCAN_SUMMARY="))
                self.log.emit(line)
            elif line.startswith("LOCAL_LIBRARY_SUMMARY="):
                self._summary = json.loads(line.removeprefix("LOCAL_LIBRARY_SUMMARY="))
                self.log.emit(line)
            elif line:
                self.log.emit(line)

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._read_output()
        if self._finished_before_ready(exit_code):
            return
        if exit_code == 0 and self._summary is not None:
            self.completed.emit(self._summary)
        else:
            self.failed.emit(f"Scan worker exited with code {exit_code}")


class RenderProcessRunner(_ProcessRunnerBase):
    log = Signal(str)
    progress = Signal(dict)
    completed = Signal(dict)
    failed = Signal(str)
    control_changed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._init_worker_process()
        self._buffer = ""
        self._summary: dict[str, object] | None = None

    @property
    def running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def start(
        self,
        *,
        db_path: Path | None = None,
        audio_root: Path | None = None,
        state_dir: Path | None = None,
        preset_ids: list[int] | None = None,
        workers: int = 4,
    ) -> None:
        if self.running:
            raise RuntimeError("Render worker is already running")
        self._buffer = ""
        self._summary = None
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        arguments = ["--workers", str(workers)]
        if db_path is not None:
            arguments.extend(["--db", str(db_path)])
        if audio_root is not None:
            arguments.extend(["--audio-root", str(audio_root)])
        if state_dir is not None:
            arguments.extend(["--state-dir", str(state_dir)])
        for preset_id in preset_ids or ():
            arguments.extend(["--preset-id", str(preset_id)])
        self._start_worker("render-library", arguments)

    def pause(self) -> None:
        if self.running:
            self.process.write(b"PAUSE\n")

    def resume(self) -> None:
        if self.running:
            self.process.write(b"RESUME\n")

    def cancel(self) -> None:
        if self.running:
            self.process.write(b"CANCEL\n")

    def _read_output(self) -> None:
        self._buffer += bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        while (line := self._pop_line()) is not None:
            if self._handle_worker_line(line):
                continue
            if line.startswith("RENDER_PROGRESS="):
                detail = json.loads(line.removeprefix("RENDER_PROGRESS="))
                self.progress.emit(detail)
            elif line.startswith("RENDER_SUMMARY="):
                self._summary = json.loads(line.removeprefix("RENDER_SUMMARY="))
                self.log.emit(line)
            elif line.startswith("RENDER_CONTROL="):
                state = line.removeprefix("RENDER_CONTROL=")
                self.control_changed.emit(state)
                self.log.emit(line)
            elif line:
                self.log.emit(line)

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._read_output()
        if self._finished_before_ready(exit_code):
            return
        if self._summary is not None and exit_code in (0, 130):
            self.completed.emit(self._summary)
        else:
            self.failed.emit(f"Render worker exited with code {exit_code}")


class StorageProcessRunner(_ProcessRunnerBase):
    """Run cross-volume migration/cleanup without blocking Qt's UI thread."""

    log = Signal(str)
    progress = Signal(dict)
    completed = Signal(dict)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._init_worker_process()
        self._result: dict[str, object] | None = None
        self._error: str | None = None

    @property
    def running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def start(self, arguments: list[str]) -> None:
        if self.running:
            raise RuntimeError("A storage operation is already running")
        self._buffer = ""
        self._result = None
        self._error = None
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        self._start_worker("storage", arguments)

    def _read_output(self) -> None:
        self._buffer += bytes(self.process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        while (line := self._pop_line()) is not None:
            if self._handle_worker_line(line):
                continue
            if line.startswith("STORAGE_PROGRESS="):
                self.progress.emit(json.loads(line.split("=", 1)[1]))
            elif line.startswith("STORAGE_RESULT="):
                self._result = json.loads(line.split("=", 1)[1])
            elif line.startswith("STORAGE_ERROR="):
                self._error = line.split("=", 1)[1]
            if line:
                self.log.emit(line)

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._read_output()
        if self._finished_before_ready(exit_code):
            return
        if exit_code == 0 and self._result is not None:
            self.completed.emit(self._result)
        else:
            self.failed.emit(
                self._error or f"Storage worker exited with code {exit_code}"
            )


class UpdateCheckProcessRunner(_ProcessRunnerBase):
    """Read-only private-release check; never blocks the UI thread."""

    log = Signal(str)
    completed = Signal(dict)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._init_worker_process()
        self._buffer = ""
        self._result: dict[str, object] | None = None

    @property
    def running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def start(self) -> None:
        if self.running:
            raise RuntimeError("An update check is already running")
        self._buffer = ""
        self._result = None
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        self._start_worker("check-update", [])

    def _read_output(self) -> None:
        self._buffer += bytes(self.process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        while (line := self._pop_line()) is not None:
            if self._handle_worker_line(line):
                continue
            if line.startswith("UPDATE_CHECK_RESULT="):
                self._result = json.loads(line.split("=", 1)[1])
            if line:
                self.log.emit(line)

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._read_output()
        if self._finished_before_ready(exit_code):
            return
        if exit_code == 0 and self._result is not None:
            self.completed.emit(self._result)
        else:
            self.failed.emit(f"Update check exited with code {exit_code}")


class UpdateDownloadProcessRunner(_ProcessRunnerBase):
    """Download a checksum-pinned installer without freezing PatchLab."""

    log = Signal(str)
    progress = Signal(dict)
    completed = Signal(dict)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._init_worker_process()
        self._buffer = ""
        self._result: dict[str, object] | None = None

    @property
    def running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def start(self, package: dict[str, object]) -> None:
        if self.running:
            raise RuntimeError("An update download is already running")
        required = ("name", "version", "size", "sha256")
        if any(not package.get(key) for key in required):
            raise ValueError("Update package metadata is incomplete")
        self._buffer = ""
        self._result = None
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        self._start_worker(
            "download-update",
            [
                "--name", str(package["name"]),
                "--version", str(package["version"]),
                "--size", str(package["size"]),
                "--sha256", str(package["sha256"]),
            ],
        )

    def _read_output(self) -> None:
        self._buffer += bytes(self.process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        while (line := self._pop_line()) is not None:
            if self._handle_worker_line(line):
                continue
            if line.startswith("UPDATE_DOWNLOAD_PROGRESS="):
                self.progress.emit(json.loads(line.split("=", 1)[1]))
            elif line.startswith("UPDATE_DOWNLOAD_RESULT="):
                self._result = json.loads(line.split("=", 1)[1])
            if line:
                self.log.emit(line)

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._read_output()
        if self._finished_before_ready(exit_code):
            return
        if exit_code == 0 and self._result is not None:
            self.completed.emit(self._result)
        else:
            self.failed.emit(f"Update download exited with code {exit_code}")


class FactoryVerificationProcessRunner(_ProcessRunnerBase):
    """Hash installed factory files without delaying the first usable window."""

    log = Signal(str)
    completed = Signal(dict)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._init_worker_process()
        self._result: dict[str, object] | None = None

    @property
    def running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def start(self) -> None:
        if self.running:
            return
        self._buffer = ""
        self._result = None
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        self._start_worker("factory-verify", [])

    def _read_output(self) -> None:
        self._buffer += bytes(self.process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        while (line := self._pop_line()) is not None:
            if self._handle_worker_line(line):
                continue
            if line.startswith("FACTORY_VERIFY_RESULT="):
                self._result = json.loads(line.split("=", 1)[1])
            elif line:
                self.log.emit(line)

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._read_output()
        if self._finished_before_ready(exit_code):
            return
        if exit_code == 0 and self._result is not None:
            self.completed.emit(self._result)
        else:
            self.failed.emit(f"Factory preset check exited with code {exit_code}")


class BugReportProcessRunner(_ProcessRunnerBase):
    """Upload a user-approved diagnostics bundle without blocking PatchLab."""

    log = Signal(str)
    completed = Signal(dict)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._init_worker_process()
        self._result: dict[str, object] | None = None
        self._error: str | None = None
        #: Stable machine-readable reason for the last failure ("" if none).
        self.error_code: str = ""
        #: The saved report being (re)sent; a retry re-sends the same saved files.
        self.request_path: Path | None = None

    @property
    def running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def start(self, request_path: Path) -> None:
        if self.running:
            raise RuntimeError("A bug report is already being sent")
        self._buffer = ""
        self._result = None
        self._error = None
        self.error_code = ""
        self.request_path = Path(request_path)
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        self._start_worker("bug-report", ["--request", str(request_path)])

    def _read_output(self) -> None:
        self._buffer += bytes(self.process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        while (line := self._pop_line()) is not None:
            if self._handle_worker_line(line):
                continue
            if line.startswith("BUG_REPORT_RESULT="):
                self._result = json.loads(line.split("=", 1)[1])
            elif line.startswith("BUG_REPORT_ERROR_CODE="):
                self.error_code = line.split("=", 1)[1].strip()
            elif line.startswith("BUG_REPORT_ERROR="):
                self._error = line.split("=", 1)[1]
            elif line:
                self.log.emit(line)

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._read_output()
        if self._finished_before_ready(exit_code):
            return
        if exit_code == 0 and self._result is not None:
            self.completed.emit(self._result)
        else:
            if not self.error_code:
                self.error_code = "worker_failed"
            self.failed.emit(
                self._error or "The upload didn't complete."
            )


class AnalyzeProcessRunner(_ProcessRunnerBase):
    log = Signal(str)
    progress = Signal(dict)
    completed = Signal(dict)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._init_worker_process()
        self._buffer = ""
        self._summary: dict[str, object] | None = None
        self._phase = "starting"

    @property
    def running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def start(
        self,
        deep_training: bool,
        *,
        smoke_db: Path | None = None,
        smoke_feature_dir: Path | None = None,
    ) -> None:
        if self.running:
            raise RuntimeError("Analyze worker is already running")
        self._buffer = ""
        self._summary = None
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        arguments: list[str] = []
        if deep_training:
            arguments.append("--deep-training")
        if smoke_db is not None:
            if smoke_feature_dir is None:
                raise ValueError("smoke_feature_dir is required with smoke_db")
            arguments.extend(
                [
                    "--packaged-smoke-db",
                    str(smoke_db),
                    "--packaged-smoke-feature-dir",
                    str(smoke_feature_dir),
                ]
            )
        self._start_worker("analyze", arguments)

    def cancel(self) -> None:
        if self.running:
            self.process.terminate()

    def _read_output(self) -> None:
        self._buffer += bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        while (line := self._pop_line()) is not None:
            if self._handle_worker_line(line):
                continue
            if line.startswith("MILESTONE3_PHASE="):
                self._phase = line.split("=", 1)[1]
                self.progress.emit({"phase": self._phase})
            elif line.startswith("ANALYZE_PROGRESS="):
                detail = json.loads(line.split("=", 1)[1])
                detail["phase"] = "embeddings"
                self.progress.emit(detail)
            elif line.startswith("SYNTHETIC_PROGRESS="):
                detail = json.loads(line.split("=", 1)[1])
                detail["phase"] = "synthetic-serum1"
                self.progress.emit(detail)
            elif line.startswith("TRAIN_PROGRESS="):
                detail = json.loads(line.split("=", 1)[1])
                detail["phase"] = "training"
                self.progress.emit(detail)
            elif line.startswith("MILESTONE3_SUMMARY="):
                self._summary = json.loads(line.split("=", 1)[1])
            if line:
                self.log.emit(line)

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._read_output()
        if self._finished_before_ready(exit_code):
            return
        if exit_code == 0 and self._summary is not None:
            self.completed.emit(self._summary)
        else:
            self.failed.emit(f"Analyze & Learn exited with code {exit_code} during {self._phase}")


#: Default inactivity budget for the Match worker, measured from its last
#: progress line. Generous on purpose: this is not a time limit on matching, it
#: is the guarantee that a worker producing *nothing* still terminates instead
#: of leaving the UI spinning. The in-worker phase-aware stall detector
#: (core/operation_state.py) normally fires long before this does; this exists
#: because the reported hang produced no output and no exception at all, so the
#: GUI needs its own backstop that does not depend on the worker being healthy
#: enough to report anything.
DEFAULT_MATCH_INACTIVITY_TIMEOUT_MS = 20 * 60 * 1000


class EngineConnection(_ProcessRunnerBase):
    """The GUI's side of the warm engine (``scripts.engine_server``).

    One long-lived worker process serves Match and export jobs one at a time,
    keeping the CLAP model, the parameter models and the Serum render workers
    loaded between them. Jobs are routed back to the runner that submitted them
    (``MatchProcessRunner`` / ``ExportProcessRunner``), which parse exactly the
    same ``MATCH_*`` / ``EXPORT_*`` lines as before.

    This is an optimisation, never a dependency: if the engine cannot start,
    dies, or is disabled (``PATCHLAB_WARM_ENGINE=0``), the runners fall back to
    their original one-process-per-job behaviour. Two unexpected engine deaths
    in a session turn it off for the rest of that session.
    """

    log = Signal(str)
    failed = Signal(str)

    #: Unexpected engine deaths tolerated before falling back to one-shot workers.
    MAX_FAILURES = 2

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._init_worker_process()
        self._queue: list[tuple[str, str, list[str], QObject | None, str]] = []
        self._active: tuple[str, QObject | None] | None = None
        self._next_id = 0
        self._failures = 0
        self._cancelled = False
        self._starting = False
        self._warm_requested = ""
        self._idle_exit_seen = False
        self._active_output_seen = False
        self._active_job: tuple[str, list[str], str] | None = None
        self._disabled = os.environ.get("PATCHLAB_WARM_ENGINE", "1").strip() == "0"

    # -- state ----------------------------------------------------------

    @property
    def usable(self) -> bool:
        return not self._disabled and self._failures < self.MAX_FAILURES

    @property
    def process_running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    @property
    def ready(self) -> bool:
        return self.process_running and self._worker_ready

    @property
    def busy(self) -> bool:
        return self._active is not None or bool(self._queue)

    def pid(self) -> int:
        return int(self.process.processId())

    def is_running_kind(self, kind: str) -> bool:
        """Whether a job of ``kind`` is running or waiting in the engine."""

        return (
            self._active is not None and self._active_job is not None and self._active_job[0] == kind
        ) or any(item[1] == kind for item in self._queue)

    def has_job(self, runner: QObject) -> bool:
        """Whether ``runner`` still has a job running or waiting in this engine."""

        return (self._active is not None and self._active[1] is runner) or any(
            item[3] is runner for item in self._queue
        )

    # -- submitting work ------------------------------------------------

    def submit(self, runner: QObject | None, kind: str, argv: list[str]) -> None:
        self._next_id += 1
        # Each job gets its own correlation id (the engine outlives any single
        # job), so a support bundle can still tell one Match from the next.
        operation_id = ""
        try:
            from core.diagnostics import new_operation_id

            operation_id = new_operation_id(kind)
        except Exception:
            pass
        if runner is not None:
            runner.operation_id = operation_id
        self._queue.append((str(self._next_id), kind, list(argv), runner, operation_id))
        self._pump()

    def warm(self, target_synth: str) -> None:
        """Load models and render workers ahead of the first Match."""

        if not self.usable or self.busy or self._warm_requested == target_synth and self.process_running:
            return
        self._warm_requested = target_synth
        self.submit(None, "warm", ["--target-synth", target_synth])

    def cancel(self, runner: QObject) -> None:
        """Cancel ``runner``'s job: drop it if queued, restart the engine if running."""

        self._queue = [item for item in self._queue if item[3] is not runner]
        if self._active is not None and self._active[1] is runner:
            self._cancelled = True
            self.process.terminate()

    def stop(self) -> None:
        """Ask the engine to exit (application shutdown)."""

        if not self.process_running:
            return
        try:
            self.process.write(b'{"id": "0", "kind": "shutdown"}\n')
            self.process.closeWriteChannel()
        except Exception:
            pass
        if not self.process.waitForFinished(3000):
            self.process.kill()

    # -- plumbing -------------------------------------------------------

    def _pump(self) -> None:
        if self._active is not None or not self._queue:
            return
        if not self.process_running:
            if not self.usable:
                self._release_queue_to_fallback()
                return
            self._starting = True
            self._cancelled = False
            self._buffer = ""
            self.process.setWorkingDirectory(str(PROJECT_ROOT))
            self._start_worker("engine", [])
            return
        if not self._worker_ready:
            return  # sent as soon as the handshake arrives
        job_id, kind, argv, runner, operation_id = self._queue.pop(0)
        self._active = (job_id, runner)
        self._active_job = (kind, argv, operation_id)
        self._active_output_seen = False
        payload = json.dumps(
            {"id": job_id, "kind": kind, "argv": argv, "operation_id": operation_id}
        )
        self.process.write(payload.encode("utf-8") + b"\n")

    def _release_queue_to_fallback(self) -> None:
        queued, self._queue = self._queue, []
        for _job_id, kind, argv, runner, _operation_id in queued:
            fallback = getattr(runner, "_engine_fallback", None)
            if fallback is not None:
                fallback(kind, argv)

    def _handle_worker_line(self, line: str) -> bool:
        handled = super()._handle_worker_line(line)
        if handled and self._worker_ready:
            self._starting = False
            self._pump()
        return handled

    def _read_output(self) -> None:
        self._buffer += bytes(self.process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        while (line := self._pop_line()) is not None:
            if self._handle_worker_line(line):
                continue
            if line.startswith("ENGINE_IDLE_EXIT="):
                self._idle_exit_seen = True
                continue
            if self._active is not None:
                self._active_output_seen = True
            if line.startswith("ENGINE_JOB_DONE="):
                try:
                    exit_code = int(json.loads(line.split("=", 1)[1]).get("exit_code", 1))
                except (ValueError, TypeError):
                    exit_code = 1
                active, self._active = self._active, None
                if active is not None and active[1] is not None:
                    done = getattr(active[1], "_engine_done", None)
                    if done is not None:
                        done(exit_code)
                self._pump()
                continue
            active_runner = self._active[1] if self._active is not None else None
            handler = getattr(active_runner, "_engine_line", None)
            if handler is not None:
                handler(line)
            elif line:
                self.log.emit(line)

    def _emit_startup_failure(self, message: str) -> None:
        # The engine never started: count it, tell the queued jobs to run the
        # one-shot way, and keep the application working.
        if self._startup_failure_emitted:
            return
        self._startup_failure_emitted = True
        self._starting = False
        self._failures += 1
        self.log.emit(f"Warm engine unavailable ({message}); using one-shot workers")
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()
        self._fail_active("The PatchLab engine could not start.")
        self._release_queue_to_fallback()

    def _fail_active(self, message: str) -> None:
        active, self._active = self._active, None
        if active is not None and active[1] is not None:
            died = getattr(active[1], "_engine_done", None)
            if died is not None:
                died(None, message)

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._read_output()
        self._startup_timer.stop()
        was_cancelled, self._cancelled = self._cancelled, False
        was_ready, self._worker_ready = self._worker_ready, False
        idle_exit, self._idle_exit_seen = self._idle_exit_seen, False
        if self._startup_failure_emitted:
            self._startup_failure_emitted = False
            return
        if not was_ready and self._starting:
            self._emit_startup_failure(f"engine exited with code {exit_code} before its handshake")
            self._startup_failure_emitted = False
            return
        if self._active is not None:
            if idle_exit and not self._active_output_seen and self._active[1] is not None:
                # The engine chose to exit on its idle timeout at the instant a
                # job arrived; nothing ran, so run the job on a fresh engine.
                job_id, runner = self._active
                self._active = None
                retry = getattr(self, "_active_job", None)
                if retry is not None:
                    self._queue.insert(0, (job_id, retry[0], retry[1], runner, retry[2]))
            else:
                if not was_cancelled:
                    self._failures += 1
                self._fail_active(
                    "Match stopped." if was_cancelled
                    else "PatchLab's engine stopped unexpectedly. Your selected audio is unchanged — you can try again."
                )
        # Work queued while the engine was shutting down restarts it.
        self._pump()


class MatchProcessRunner(_ProcessRunnerBase):
    log = Signal(str)
    progress = Signal(dict)
    completed = Signal(str)
    failed = Signal(str)
    stalled = Signal(dict)

    #: Class-level defaults: a runner that never received an engine runs one-shot.
    _engine: "EngineConnection | None" = None
    _engine_job = False

    def __init__(
        self, parent: QObject | None = None, engine: "EngineConnection | None" = None
    ) -> None:
        super().__init__(parent)
        self._init_worker_process()
        #: The shared warm engine, if any. Jobs run there when it is usable and
        #: fall back to a one-shot worker process otherwise.
        self._engine = engine
        self._engine_job = False
        self._buffer = ""
        self._result: str | None = None
        self._error: str | None = None
        self._last_phase = ""
        self._progress_lines = 0
        self._stall_reported = False
        # Liveness watchdog. Restarted by every worker output line, so a worker
        # that is slow but talking is never killed, while one that has gone
        # silent is turned into a clean terminal failure.
        self._liveness_timer = QTimer(self)
        self._liveness_timer.setSingleShot(True)
        self._liveness_timer.timeout.connect(self._liveness_timed_out)

    @property
    def running(self) -> bool:
        return self._engine_job or self.process.state() != QProcess.ProcessState.NotRunning

    def _liveness_interval_ms(self) -> int:
        try:
            return max(
                1_000,
                int(
                    os.environ.get(
                        "PATCHLAB_MATCH_INACTIVITY_TIMEOUT_MS",
                        str(DEFAULT_MATCH_INACTIVITY_TIMEOUT_MS),
                    )
                ),
            )
        except ValueError:
            return DEFAULT_MATCH_INACTIVITY_TIMEOUT_MS

    def _touch_liveness(self) -> None:
        if self.running:
            self._liveness_timer.start(self._liveness_interval_ms())

    def _liveness_timed_out(self) -> None:
        """No worker output for the whole inactivity budget: recover the UI.

        Evidence first, then recovery: the worker is asked to dump its own state
        via the flight recorder's postmortem file before it is killed, then the
        UI receives a terminal `failed` so controls come back.
        """

        if not self.running or self._stall_reported:
            return
        self._stall_reported = True
        seconds = self._liveness_interval_ms() / 1000.0
        detail = {
            "reason": (
                f"the match worker produced no output for {seconds:.0f}s during "
                f"phase {self._last_phase or 'unknown'}"
            ),
            "last_phase": self._last_phase,
            "progress_lines": self._progress_lines,
            "pid": self._engine.pid() if self._engine_job else int(self.process.processId()),
            "inactivity_seconds": seconds,
        }
        try:
            from core.diagnostics import record
            from core.operation_state import capture_thread_stacks, write_postmortem

            record(
                "match-runner",
                "stall_detected",
                detail["reason"],
                severity="error",
                phase=self._last_phase,
                decision_reason=(
                    "GUI-side liveness backstop: a worker that reports nothing for a "
                    "whole inactivity budget is terminated so the UI cannot stay in a "
                    "loading state forever"
                ),
                **detail,
            )
            write_postmortem(
                {
                    "trigger": "stall",
                    "operation": "match",
                    "source": "gui-liveness-watchdog",
                    "current_phase": self._last_phase,
                    "seconds_since_last_progress": seconds,
                    "stall_budget_seconds": seconds,
                    "detail": detail,
                    "thread_stacks": capture_thread_stacks(),
                }
            )
        except Exception:
            pass
        self.stalled.emit(detail)
        self.log.emit(f"Match stopped: {detail['reason']}")
        if self._engine_job:
            self._engine.process.kill()
        else:
            self.process.kill()
        self.failed.emit(
            "PatchLab stopped waiting because the match made no progress. "
            "Your selected audio is unchanged — you can try again."
        )

    def start(
        self,
        audio: Path,
        *,
        target_synth: str,
        budget: str,
        offset: float,
        session_root: Path | None = None,
        factory_only: bool = False,
        factory_mapping: Path | None = None,
        local_db: Path | None = None,
        local_audio_root: Path | None = None,
    ) -> None:
        if self.running:
            raise RuntimeError("A sound match is already running")
        self._buffer = ""
        self._result = None
        self._error = None
        self._last_phase = ""
        self._progress_lines = 0
        self._stall_reported = False
        arguments = [
            str(audio),
            "--target-synth",
            target_synth,
            "--budget",
            budget,
            "--offset",
            str(offset),
        ]
        if session_root is not None:
            arguments.extend(["--session-root", str(session_root)])
        if factory_only:
            arguments.append("--factory-only")
        if factory_mapping is not None:
            arguments.extend(["--factory-mapping", str(factory_mapping)])
        if local_db is not None:
            arguments.extend(["--local-db", str(local_db)])
        if local_audio_root is not None:
            arguments.extend(["--local-audio-root", str(local_audio_root)])
        if self._engine is not None and self._engine.usable and not factory_only:
            self._engine_job = True
            self._engine.submit(self, "match", arguments)
            self._touch_liveness()
            return
        self._start_one_shot(arguments)

    def _start_one_shot(self, arguments: list[str]) -> None:
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        self._start_worker("match", arguments)
        self._touch_liveness()

    def cancel(self) -> None:
        if self._engine_job:
            self._engine.cancel(self)
            if not self._engine.has_job(self):
                # It was only queued, so no completion will ever arrive.
                self._engine_job = False
                self._liveness_timer.stop()
        elif self.running:
            self.process.terminate()

    def _consume_line(self, line: str) -> None:
        # Any output at all is evidence of life, so the inactivity budget
        # restarts here rather than only on structured progress.
        self._touch_liveness()
        if line.startswith("MATCH_PROGRESS="):
            detail = json.loads(line.split("=", 1)[1])
            self._progress_lines += 1
            self._last_phase = str(detail.get("phase", "")) or self._last_phase
            self.progress.emit(detail)
        elif line.startswith("MATCH_RESULT="):
            self._result = line.split("=", 1)[1]
        elif line.startswith("MATCH_ERROR="):
            self._error = line.split("=", 1)[1]
        if line:
            self.log.emit(line)

    def _read_output(self) -> None:
        self._buffer += bytes(self.process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        while (line := self._pop_line()) is not None:
            self._touch_liveness()
            if self._handle_worker_line(line):
                continue
            self._consume_line(line)

    # -- warm engine callbacks (see EngineConnection) ---------------------

    def _engine_line(self, line: str) -> None:
        self._consume_line(line)

    def _engine_done(self, exit_code: int | None, message: str = "") -> None:
        self._engine_job = False
        self._liveness_timer.stop()
        if self._stall_reported:
            return  # the watchdog already reported the terminal failure
        if exit_code == 0 and self._result:
            self.completed.emit(self._result)
        else:
            self.failed.emit(
                self._error or message or f"Match worker exited with code {exit_code}"
            )

    def _engine_fallback(self, kind: str, argv: list[str]) -> None:
        """The engine is unavailable: run this job as a one-shot worker instead."""

        self._engine_job = False
        self._start_one_shot(argv)

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._liveness_timer.stop()
        self._read_output()
        if self._finished_before_ready(exit_code):
            return
        if self._stall_reported:
            # The watchdog already emitted a terminal failure; do not emit a
            # second, less useful one for the kill it caused.
            return
        if exit_code == 0 and self._result:
            self.completed.emit(self._result)
        else:
            self.failed.emit(self._error or f"Match worker exited with code {exit_code}")


class ExportProcessRunner(_ProcessRunnerBase):
    log = Signal(str)
    completed = Signal(dict)
    failed = Signal(str)

    #: Class-level defaults: a runner that never received an engine runs one-shot.
    _engine: "EngineConnection | None" = None
    _engine_job = False

    def __init__(
        self, parent: QObject | None = None, engine: "EngineConnection | None" = None
    ) -> None:
        super().__init__(parent)
        self._init_worker_process()
        #: See MatchProcessRunner: the shared warm engine, with one-shot fallback.
        self._engine = engine
        self._engine_job = False
        self._buffer = ""
        self._result: dict | None = None
        self._error: str | None = None
        #: The structured EXPORT_FAILURE record of the last failed incident save.
        self.failure: dict | None = None
        #: The save incident the running (or last) export belongs to, if any.
        self.incident_id: str | None = None

    @property
    def running(self) -> bool:
        return self._engine_job or self.process.state() != QProcess.ProcessState.NotRunning

    def start(
        self,
        result_path: Path,
        output_path: Path,
        *,
        existing_match: int | None = None,
        incident_id: str | None = None,
        trigger: str = "auto",
        attempts: int = 1,
    ) -> None:
        if self.running:
            raise RuntimeError("A preset export is already running")
        self._buffer = ""
        self._result = None
        self._error = None
        self.failure = None
        self.incident_id = incident_id
        arguments = [str(result_path), str(output_path)]
        if existing_match is not None:
            # One export implementation for both the generated recommendation and
            # a closest match; only the selector differs.
            arguments += ["--existing-match", str(int(existing_match))]
        if incident_id:
            # The generated-preset save lifecycle: bounded retries, classified
            # failures, one incident record (core.preset_save).
            arguments += [
                "--incident", str(incident_id),
                "--trigger", str(trigger),
                "--attempts", str(max(1, int(attempts))),
            ]
        # A Match can take minutes; a save must not wait behind it (as one-shot
        # exports never did), so it only joins the engine when no Match is there.
        if (
            self._engine is not None
            and self._engine.usable
            and not self._engine.is_running_kind("match")
        ):
            self._engine_job = True
            self._engine.submit(self, "export", arguments)
            return
        self._start_one_shot(arguments)

    def _start_one_shot(self, arguments: list[str]) -> None:
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        self._start_worker("export", arguments)

    def cancel(self) -> None:
        if self._engine_job:
            self._engine.cancel(self)
            if not self._engine.has_job(self):
                self._engine_job = False
        elif self.running:
            self.process.terminate()

    def _consume_line(self, line: str) -> None:
        if line.startswith("EXPORT_RESULT="):
            self._result = json.loads(line.split("=", 1)[1])
        elif line.startswith("EXPORT_ERROR="):
            self._error = line.split("=", 1)[1]
        elif line.startswith("EXPORT_FAILURE="):
            try:
                self.failure = json.loads(line.split("=", 1)[1])
            except ValueError:
                self.failure = None
        if line:
            self.log.emit(line)

    def _read_output(self) -> None:
        self._buffer += bytes(self.process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        while (line := self._pop_line()) is not None:
            if self._handle_worker_line(line):
                continue
            self._consume_line(line)

    # -- warm engine callbacks (see EngineConnection) ---------------------

    def _engine_line(self, line: str) -> None:
        self._consume_line(line)

    def _engine_done(self, exit_code: int | None, message: str = "") -> None:
        self._engine_job = False
        if exit_code == 0 and self._result is not None:
            self.completed.emit(self._result)
        else:
            self.failed.emit(
                self._error or message or f"Export worker exited with code {exit_code}"
            )

    def _engine_fallback(self, kind: str, argv: list[str]) -> None:
        self._engine_job = False
        self._start_one_shot(argv)

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._read_output()
        if self._finished_before_ready(exit_code):
            return
        if exit_code == 0 and self._result is not None:
            self.completed.emit(self._result)
        else:
            self.failed.emit(self._error or f"Export worker exited with code {exit_code}")


class PreviewProcessRunner(_ProcessRunnerBase):
    log = Signal(str)
    completed = Signal(str)
    failed = Signal(str)
    request_completed = Signal(str, str)
    request_failed = Signal(str, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._init_worker_process()
        self._buffer = ""
        self._result: str | None = None
        self._error: str | None = None
        self._startup_error: str | None = None
        self._queue: list[dict[str, object]] = []
        self._active: dict[str, object] | None = None

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    def start(
        self,
        source: Path,
        *,
        synth: str,
        midi_note: int,
        content_hash: str,
        output_root: Path | None = None,
    ) -> str:
        request_id = uuid.uuid4().hex
        self._queue.append(
            {
                "id": request_id,
                "worker": "factory-preview",
                "source": Path(source),
                "synth": synth,
                "midi_note": int(midi_note),
                "content_hash": content_hash,
                "output_root": Path(output_root) if output_root is not None else None,
            }
        )
        self._launch_next()
        return request_id

    def start_recommendation(
        self,
        result_path: Path,
        midi_note: int,
        *,
        output_root: Path | None = None,
        cache_key: str | None = None,
    ) -> str:
        request_id = uuid.uuid4().hex
        self._queue.append(
            {
                "id": request_id,
                "worker": "recommendation-preview",
                "result_path": Path(result_path),
                "midi_note": int(midi_note),
                "output_root": Path(output_root) if output_root is not None else None,
                "cache_key": cache_key,
            }
        )
        self._launch_next()
        return request_id

    def _launch_next(self) -> None:
        if (
            self._active is not None
            or self.process.state() != QProcess.ProcessState.NotRunning
            or not self._queue
        ):
            return
        self._active = self._queue.pop(0)
        self._buffer = ""
        self._result = None
        self._error = None
        self._startup_error = None
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        worker = str(self._active["worker"])
        if worker == "factory-preview":
            arguments = [
                str(self._active["source"]),
                "--synth",
                str(self._active["synth"]),
                "--note",
                str(self._active["midi_note"]),
                "--content-hash",
                str(self._active["content_hash"]),
            ]
        else:
            arguments = [
                str(self._active["result_path"]),
                "--note",
                str(self._active["midi_note"]),
            ]
            cache_key = self._active.get("cache_key")
            if cache_key:
                arguments.extend(["--cache-key", str(cache_key)])
        output_root = self._active.get("output_root")
        if output_root is not None:
            arguments.extend(["--output-root", str(output_root)])
        self._start_worker(worker, arguments)

    def cancel(self) -> None:
        while self._queue:
            queued = self._queue.pop(0)
            self.request_failed.emit(str(queued["id"]), "Preview request cancelled")
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.terminate()

    def _emit_startup_failure(self, message: str) -> None:
        self._startup_error = message
        super()._emit_startup_failure(message)

    def _read_output(self) -> None:
        self._buffer += bytes(self.process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        while (line := self._pop_line()) is not None:
            if self._handle_worker_line(line):
                continue
            if line.startswith("PREVIEW_RESULT="):
                self._result = str(json.loads(line.split("=", 1)[1])["path"])
            elif line.startswith("PREVIEW_ERROR="):
                self._error = line.split("=", 1)[1]
            if line:
                self.log.emit(line)

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._read_output()
        active = self._active
        request_id = str(active["id"]) if active is not None else ""
        if self._finished_before_ready(exit_code):
            if request_id:
                self.request_failed.emit(
                    request_id,
                    self._startup_error
                    or f"Preview worker exited with code {exit_code} before startup",
                )
            self._active = None
            QTimer.singleShot(0, self._launch_next)
            return
        if exit_code == 0 and self._result:
            self.completed.emit(self._result)
            if request_id:
                self.request_completed.emit(request_id, self._result)
        else:
            error = self._error or f"Preview worker exited with code {exit_code}"
            self.failed.emit(error)
            if request_id:
                self.request_failed.emit(request_id, error)
        self._active = None
        QTimer.singleShot(0, self._launch_next)


def worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", type=Path)
    return parser


def worker_main() -> int:
    args = worker_parser().parse_args()
    if args.scan is None:
        return 2
    from core.preset_scan import scan_and_ingest

    def progress(current: int, total: int) -> None:
        print(f"WORKER_PROGRESS={current}/{total}", flush=True)

    scan_and_ingest(args.scan, log=lambda message: print(message, flush=True), progress=progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(worker_main())

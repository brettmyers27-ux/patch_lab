"""QProcess-backed background jobs and worker-process entry points."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QTimer, Signal

from core.worker_runtime import (
    DEFAULT_STARTUP_TIMEOUT_MS,
    WORKER_READY_PREFIX,
    worker_invocation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
        self._worker_ready = False
        self._startup_failure_emitted = False

    def _start_worker(self, worker_name: str, arguments: list[str]) -> None:
        self._worker_name = worker_name
        self._worker_ready = False
        self._startup_failure_emitted = False
        program, invocation = worker_invocation(worker_name, arguments)
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
    completed = Signal(dict)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._init_worker_process()
        self._buffer = ""
        self._summary: dict[str, int] | None = None

    def start(self, root: Path, *, local_library: bool = False) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            raise RuntimeError("Scan worker is already running")
        self._buffer = ""
        self._summary = None
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        if local_library:
            self._start_worker(
                "local-library",
                [
                    str(root),
                    "--workers",
                    "4",
                ],
            )
        else:
            self._start_worker("scan", ["--scan", str(root)])

    def cancel(self) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.terminate()

    def _read_output(self) -> None:
        self._buffer += bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if self._handle_worker_line(line):
                continue
            if line.startswith("WORKER_PROGRESS="):
                current, total = line.removeprefix("WORKER_PROGRESS=").split("/", 1)
                self.progress.emit(int(current), int(total))
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
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
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
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
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


class MatchProcessRunner(_ProcessRunnerBase):
    log = Signal(str)
    progress = Signal(dict)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._init_worker_process()
        self._buffer = ""
        self._result: str | None = None
        self._error: str | None = None

    @property
    def running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

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
    ) -> None:
        if self.running:
            raise RuntimeError("A sound match is already running")
        self._buffer = ""
        self._result = None
        self._error = None
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
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
        self._start_worker("match", arguments)

    def cancel(self) -> None:
        if self.running:
            self.process.terminate()

    def _read_output(self) -> None:
        self._buffer += bytes(self.process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if self._handle_worker_line(line):
                continue
            if line.startswith("MATCH_PROGRESS="):
                self.progress.emit(json.loads(line.split("=", 1)[1]))
            elif line.startswith("MATCH_RESULT="):
                self._result = line.split("=", 1)[1]
            elif line.startswith("MATCH_ERROR="):
                self._error = line.split("=", 1)[1]
            if line:
                self.log.emit(line)

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._read_output()
        if self._finished_before_ready(exit_code):
            return
        if exit_code == 0 and self._result:
            self.completed.emit(self._result)
        else:
            self.failed.emit(self._error or f"Match worker exited with code {exit_code}")


class ExportProcessRunner(_ProcessRunnerBase):
    log = Signal(str)
    completed = Signal(dict)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._init_worker_process()
        self._buffer = ""
        self._result: dict | None = None
        self._error: str | None = None

    @property
    def running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def start(self, result_path: Path, output_path: Path) -> None:
        if self.running:
            raise RuntimeError("A preset export is already running")
        self._buffer = ""
        self._result = None
        self._error = None
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        self._start_worker(
            "export",
            [
                str(result_path),
                str(output_path),
            ],
        )

    def cancel(self) -> None:
        if self.running:
            self.process.terminate()

    def _read_output(self) -> None:
        self._buffer += bytes(self.process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if self._handle_worker_line(line):
                continue
            if line.startswith("EXPORT_RESULT="):
                self._result = json.loads(line.split("=", 1)[1])
            elif line.startswith("EXPORT_ERROR="):
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
            self.failed.emit(self._error or f"Export worker exited with code {exit_code}")


class PreviewProcessRunner(_ProcessRunnerBase):
    log = Signal(str)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._init_worker_process()
        self._buffer = ""
        self._result: str | None = None
        self._error: str | None = None

    def start(
        self,
        source: Path,
        *,
        synth: str,
        midi_note: int,
        content_hash: str,
        output_root: Path | None = None,
    ) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            raise RuntimeError("A factory preview is already rendering")
        self._buffer = ""
        self._result = None
        self._error = None
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        arguments = [
            str(source),
            "--synth",
            synth,
            "--note",
            str(midi_note),
            "--content-hash",
            content_hash,
        ]
        if output_root is not None:
            arguments.extend(["--output-root", str(output_root)])
        self._start_worker("factory-preview", arguments)

    def start_recommendation(self, result_path: Path, midi_note: int) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            raise RuntimeError("A recommendation preview is already rendering")
        self._buffer = ""
        self._result = None
        self._error = None
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        self._start_worker(
            "recommendation-preview",
            [
                str(result_path),
                "--note",
                str(midi_note),
            ],
        )

    def cancel(self) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.terminate()

    def _read_output(self) -> None:
        self._buffer += bytes(self.process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
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
        if self._finished_before_ready(exit_code):
            return
        if exit_code == 0 and self._result:
            self.completed.emit(self._result)
        else:
            self.failed.emit(
                self._error or f"Factory preview worker exited with code {exit_code}"
            )


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

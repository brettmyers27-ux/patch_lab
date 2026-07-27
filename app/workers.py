"""QProcess-backed background jobs and worker-process entry points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, Signal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class ScanProcessRunner(QObject):
    log = Signal(str)
    progress = Signal(int, int)
    completed = Signal(dict)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self._buffer = ""
        self._summary: dict[str, int] | None = None

    def start(self, root: Path, *, local_library: bool = False) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            raise RuntimeError("Scan worker is already running")
        self._buffer = ""
        self._summary = None
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        if local_library:
            self.process.start(
                sys.executable,
                [
                    str(PROJECT_ROOT / "scripts" / "process_local_library.py"),
                    str(root),
                    "--workers",
                    "4",
                ],
            )
        else:
            self.process.start(
                sys.executable,
                [str(Path(__file__).resolve()), "--scan", str(root)],
            )

    def cancel(self) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.terminate()

    def _read_output(self) -> None:
        self._buffer += bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
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
        if exit_code == 0 and self._summary is not None:
            self.completed.emit(self._summary)
        else:
            self.failed.emit(f"Scan worker exited with code {exit_code}")


class RenderProcessRunner(QObject):
    log = Signal(str)
    progress = Signal(dict)
    completed = Signal(dict)
    failed = Signal(str)
    control_changed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self._buffer = ""
        self._summary: dict[str, object] | None = None

    @property
    def running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def start(self) -> None:
        if self.running:
            raise RuntimeError("Render worker is already running")
        self._buffer = ""
        self._summary = None
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        script = PROJECT_ROOT / "scripts" / "render_library.py"
        self.process.start(sys.executable, [str(script), "--workers", "4"])

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
        if self._summary is not None and exit_code in (0, 130):
            self.completed.emit(self._summary)
        else:
            self.failed.emit(f"Render worker exited with code {exit_code}")


class AnalyzeProcessRunner(QObject):
    log = Signal(str)
    progress = Signal(dict)
    completed = Signal(dict)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self._buffer = ""
        self._summary: dict[str, object] | None = None
        self._phase = "starting"

    @property
    def running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def start(self, deep_training: bool) -> None:
        if self.running:
            raise RuntimeError("Analyze worker is already running")
        self._buffer = ""
        self._summary = None
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        arguments = [str(PROJECT_ROOT / "scripts" / "run_milestone3.py")]
        if deep_training:
            arguments.append("--deep-training")
        self.process.start(sys.executable, arguments)

    def cancel(self) -> None:
        if self.running:
            self.process.terminate()

    def _read_output(self) -> None:
        self._buffer += bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
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
        if exit_code == 0 and self._summary is not None:
            self.completed.emit(self._summary)
        else:
            self.failed.emit(f"Analyze & Learn exited with code {exit_code} during {self._phase}")


class MatchProcessRunner(QObject):
    log = Signal(str)
    progress = Signal(dict)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
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
            str(PROJECT_ROOT / "scripts" / "match_sound.py"),
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
        self.process.start(sys.executable, arguments)

    def cancel(self) -> None:
        if self.running:
            self.process.terminate()

    def _read_output(self) -> None:
        self._buffer += bytes(self.process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
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
        if exit_code == 0 and self._result:
            self.completed.emit(self._result)
        else:
            self.failed.emit(self._error or f"Match worker exited with code {exit_code}")


class ExportProcessRunner(QObject):
    log = Signal(str)
    completed = Signal(dict)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
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
        self.process.start(
            sys.executable,
            [
                str(PROJECT_ROOT / "scripts" / "export_match.py"),
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
            if line.startswith("EXPORT_RESULT="):
                self._result = json.loads(line.split("=", 1)[1])
            elif line.startswith("EXPORT_ERROR="):
                self._error = line.split("=", 1)[1]
            if line:
                self.log.emit(line)

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._read_output()
        if exit_code == 0 and self._result is not None:
            self.completed.emit(self._result)
        else:
            self.failed.emit(self._error or f"Export worker exited with code {exit_code}")


class PreviewProcessRunner(QObject):
    log = Signal(str)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
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
            str(PROJECT_ROOT / "scripts" / "render_factory_preview.py"),
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
        self.process.start(sys.executable, arguments)

    def start_recommendation(self, result_path: Path, midi_note: int) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            raise RuntimeError("A recommendation preview is already rendering")
        self._buffer = ""
        self._result = None
        self._error = None
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        self.process.start(
            sys.executable,
            [
                str(PROJECT_ROOT / "scripts" / "render_recommendation_preview.py"),
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
            if line.startswith("PREVIEW_RESULT="):
                self._result = str(json.loads(line.split("=", 1)[1])["path"])
            elif line.startswith("PREVIEW_ERROR="):
                self._error = line.split("=", 1)[1]
            if line:
                self.log.emit(line)

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._read_output()
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

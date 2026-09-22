"""Shared worker entry-point metadata and frozen/dev subprocess invocation."""

from __future__ import annotations

import os

import sys
from dataclasses import dataclass


WORKER_FLAG = "--patchlab-worker"
WORKER_READY_PREFIX = "PATCHLAB_WORKER_READY="
# The handshake only proves a worker reached its dispatcher, not that it
# finished loading models — but merely *spawning* a process is slow on a
# machine already saturated by render workers and multi-gigabyte model loads.
# At 20s a healthy export was killed mid-batch and then reported as a failed
# match. This is deliberately generous: the timeout exists to turn a genuinely
# hung worker into a clear error, not to police startup latency under load.
DEFAULT_STARTUP_TIMEOUT_MS = 90_000


@dataclass(frozen=True)
class WorkerEntryPoint:
    module: str
    callable_name: str = "main"


WORKER_ENTRY_POINTS: dict[str, WorkerEntryPoint] = {
    "scan": WorkerEntryPoint("app.workers", "worker_main"),
    "local-library": WorkerEntryPoint("scripts.process_local_library"),
    "process-pending": WorkerEntryPoint("scripts.process_pending_presets"),
    "factory-verify": WorkerEntryPoint("scripts.verify_factory_runtime"),
    "render-library": WorkerEntryPoint("scripts.render_library"),
    "fingerprint-local": WorkerEntryPoint("scripts.fingerprint_pending_presets"),
    "check-update": WorkerEntryPoint("scripts.check_for_update"),
    "download-update": WorkerEntryPoint("scripts.download_update"),
    "open-downloaded-update": WorkerEntryPoint("scripts.open_downloaded_update"),
    "bug-report": WorkerEntryPoint("scripts.submit_bug_report"),
    "analyze": WorkerEntryPoint("scripts.run_milestone3"),
    "match": WorkerEntryPoint("scripts.match_sound"),
    "export": WorkerEntryPoint("scripts.export_match"),
    "factory-preview": WorkerEntryPoint("scripts.render_factory_preview"),
    "recommendation-preview": WorkerEntryPoint(
        "scripts.render_recommendation_preview"
    ),
    "storage": WorkerEntryPoint("scripts.manage_storage"),
    "build-serum2-targets": WorkerEntryPoint("scripts.build_serum2_targets"),
    "analyze-library": WorkerEntryPoint("scripts.analyze_library"),
    "build-similarity-index": WorkerEntryPoint(
        "scripts.build_similarity_index"
    ),
    "generate-synthetic-serum1": WorkerEntryPoint(
        "scripts.generate_synthetic_serum1"
    ),
    "train-param-model": WorkerEntryPoint("scripts.train_param_model"),
    "roundtrip-param-model": WorkerEntryPoint(
        "scripts.roundtrip_param_model"
    ),
    "packaged-worker-gate": WorkerEntryPoint(
        "scripts.verify_packaged_workers"
    ),
    "packaged-runtime-gate": WorkerEntryPoint(
        "scripts.verify_packaged_runtime"
    ),
    "workflow-card-gate": WorkerEntryPoint(
        "scripts.verify_workflow_cards"
    ),
    "release-flow-gate": WorkerEntryPoint(
        "scripts.verify_release_flows"
    ),
    "preview-cache-gate": WorkerEntryPoint(
        "scripts.verify_preview_cache"
    ),
    "visual-redesign-gate": WorkerEntryPoint(
        "scripts.verify_visual_redesign"
    ),
    "milestone4-ui-gate": WorkerEntryPoint(
        "scripts.verify_milestone4_ui"
    ),
    "first-run-gate": WorkerEntryPoint(
        "scripts.verify_first_run"
    ),
}


SCRIPT_WORKERS = {
    "build_serum2_targets.py": "build-serum2-targets",
    "analyze_library.py": "analyze-library",
    "build_similarity_index.py": "build-similarity-index",
    "generate_synthetic_serum1.py": "generate-synthetic-serum1",
    "train_param_model.py": "train-param-model",
    "roundtrip_param_model.py": "roundtrip-param-model",
}


def is_frozen_build() -> bool:
    return bool(getattr(sys, "frozen", False))


def worker_invocation(
    worker_name: str,
    arguments: list[str] | tuple[str, ...] = (),
) -> tuple[str, list[str]]:
    """Return a QProcess/subprocess command for this runtime.

    Frozen builds re-enter the PyInstaller executable with an early dispatch
    flag. Development uses the real interpreter and the same dispatcher as a
    module, preserving identical worker behavior and the startup handshake.
    """

    if worker_name not in WORKER_ENTRY_POINTS:
        raise ValueError(f"Unknown PatchLab worker: {worker_name}")
    suffix = [worker_name, *map(str, arguments)]
    if is_frozen_build():
        return sys.executable, [WORKER_FLAG, *suffix]
    return sys.executable, ["-m", "app.worker_dispatch", *suffix]


def worker_invocation_for_script(arguments: list[str]) -> tuple[str, list[str]]:
    """Translate a Milestone 3 script command into the shared dispatcher."""

    if not arguments:
        raise ValueError("A worker script is required")
    script = arguments[0].replace("\\", "/").rsplit("/", 1)[-1]
    try:
        worker_name = SCRIPT_WORKERS[script]
    except KeyError as exc:
        raise ValueError(f"No packaged worker entry point for {script}") from exc
    return worker_invocation(worker_name, arguments[1:])


#: Exit status a worker uses when its parent closed the pipe. The GUI cancels a
#: background job by terminating the worker, so "nobody is listening any more"
#: is a normal end to the job, not a crash to report.
PARENT_GONE_EXIT = 0


def emit(line: str) -> bool:
    """Print one line to the parent, tolerating a parent that has gone away.

    A cancelled background scan used to die with an unhandled BrokenPipeError
    from inside its progress callback -- the GUI had closed the pipe on purpose,
    but the worker treated that as a crash and produced an alarming traceback.
    Returns False once the parent is gone so a caller can stop early.
    """

    import sys

    try:
        print(line, flush=True)
        return True
    except (BrokenPipeError, ValueError, OSError):
        # ValueError/OSError: the stream was already closed underneath us.
        _silence_stdout()
        return False


def _silence_stdout() -> None:
    """Point stdout at /dev/null, including its file descriptor.

    Replacing ``sys.stdout`` alone is not enough: the interpreter still flushes
    the original stream at shutdown, and on a dead pipe that turns a clean exit
    into status 120 plus an "Exception ignored" message. Re-pointing fd 1 makes
    every later write, from anywhere, harmlessly succeed.
    """

    import sys

    try:
        null = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(null, 1)
        finally:
            os.close(null)
    except Exception:
        pass
    try:
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115 - process is ending
    except Exception:
        pass

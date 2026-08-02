"""Shared worker entry-point metadata and frozen/dev subprocess invocation."""

from __future__ import annotations

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
    "render-library": WorkerEntryPoint("scripts.render_library"),
    "analyze": WorkerEntryPoint("scripts.run_milestone3"),
    "match": WorkerEntryPoint("scripts.match_sound"),
    "export": WorkerEntryPoint("scripts.export_match"),
    "factory-preview": WorkerEntryPoint("scripts.render_factory_preview"),
    "recommendation-preview": WorkerEntryPoint(
        "scripts.render_recommendation_preview"
    ),
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

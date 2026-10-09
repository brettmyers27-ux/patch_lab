#!/usr/bin/env python3
"""The warm engine: one long-lived worker that keeps Match's slow-to-build state.

Starting a Match used to cost ~9 s before any searching began (Python and ML
imports, the CLAP model, the parameter models, a pool of Serum render workers),
and saving the result cost ~8 s more because the export worker loaded CLAP and
a Serum host all over again -- once per file, so a batch repeated both for every
sound. None of that depends on the sound being matched.

This process builds it once and serves any number of jobs, one at a time, over
a line protocol. Each job runs the very same code as the one-shot ``match`` and
``export`` workers (their ``main`` functions), so results are identical and the
GUI's parsing of ``MATCH_*``/``EXPORT_*`` lines is unchanged.

Protocol
--------
stdin, one JSON object per line::

    {"id": "7", "kind": "match",    "argv": ["audio.wav", "--target-synth", "serum2", ...]}
    {"id": "8", "kind": "export",   "argv": ["result.json", "out.SerumPreset", ...]}
    {"id": "9", "kind": "warm",     "argv": ["--target-synth", "serum2"]}
    {"id": "0", "kind": "shutdown"}

stdout: whatever the job prints, then ``ENGINE_JOB_DONE={"id": ..., "exit_code": N}``.

The engine exits when its parent closes stdin, on ``shutdown``, or after
``PATCHLAB_ENGINE_IDLE_SECONDS`` (default 600) without a job, releasing the
memory the loaded models and render workers hold.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

JOB_DONE_PREFIX = "ENGINE_JOB_DONE="
DEFAULT_IDLE_SECONDS = 600.0


def idle_limit_seconds() -> float:
    try:
        return max(5.0, float(os.environ.get("PATCHLAB_ENGINE_IDLE_SECONDS", DEFAULT_IDLE_SECONDS)))
    except ValueError:
        return DEFAULT_IDLE_SECONDS


class Engine:
    """Holds the warm matcher and export verifier and runs jobs against them."""

    def __init__(self) -> None:
        from core.match_workflow import WarmMatcherCache

        self.cache = WarmMatcherCache()
        self._verifier: Any = None

    # -- jobs -----------------------------------------------------------

    def run_match(self, argv: list[str]) -> int:
        from scripts import match_sound

        return int(match_sound.main(argv, matcher_cache=self.cache) or 0)

    def verifier(self) -> Any:
        if self._verifier is None:
            from core.preset_export import PresetExportVerifier

            # Shares the matcher's CLAP model: one copy in memory, one load.
            self._verifier = PresetExportVerifier(embedder=self.cache.embedder)
        return self._verifier

    def run_export(self, argv: list[str]) -> int:
        from scripts import export_match

        try:
            return int(export_match.main(argv, verifier=self.verifier()) or 0)
        finally:
            if self._verifier is not None:
                self._verifier.reset_scratch()

    def warm(self, argv: list[str]) -> int:
        """Load everything a Match for this generation needs, before one is asked for."""

        import argparse

        from core.diagnostics import new_operation_id
        from core.platform_env import recommended_render_workers

        parser = argparse.ArgumentParser()
        parser.add_argument("--target-synth", choices=("serum1", "serum2"), default="serum2")
        parser.add_argument("--processes", type=int, default=0)
        args = parser.parse_args(argv)
        processes = args.processes or recommended_render_workers()
        matcher = self.cache.acquire(
            processes=processes,
            deterministic_render_dispatch=False,
            required_synths=(args.target_synth,),
            operation_id=new_operation_id("warm"),
        )
        self.cache.release(matcher, healthy=True)
        try:
            # Open the Serum host the export check will use, too.
            self.verifier().host(args.target_synth)
        except Exception as exc:  # warming is best-effort; the first export will retry
            print(f"ENGINE_WARM_NOTE=export host not preloaded: {type(exc).__name__}: {exc}", flush=True)
        return 0

    def dispatch(self, kind: str, argv: list[str]) -> int:
        if kind == "match":
            return self.run_match(argv)
        if kind == "export":
            return self.run_export(argv)
        if kind == "warm":
            return self.warm(argv)
        print(f"ENGINE_ERROR=unknown job kind {kind!r}", flush=True)
        return 2

    def shutdown(self) -> None:
        try:
            self.cache.close()
        except Exception:
            pass
        if self._verifier is not None:
            try:
                self._verifier.close()
            except Exception:
                pass


def _read_commands(commands: "queue.Queue[dict[str, Any] | None]") -> None:
    """Feed parsed stdin lines to the main loop; ``None`` marks end of input."""

    try:
        for raw in sys.stdin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                command = json.loads(raw)
            except ValueError:
                print(f"ENGINE_ERROR=unparseable command: {raw[:80]}", flush=True)
                continue
            if isinstance(command, dict):
                commands.put(command)
    finally:
        commands.put(None)


def _failure_line(kind: str, exc: BaseException) -> str:
    prefix = {"match": "MATCH_ERROR=", "export": "EXPORT_ERROR="}.get(kind, "ENGINE_ERROR=")
    return f"{prefix}{type(exc).__name__}: {exc}"


def main() -> int:
    commands: "queue.Queue[dict[str, Any] | None]" = queue.Queue()
    threading.Thread(target=_read_commands, args=(commands,), daemon=True).start()
    engine = Engine()
    limit = idle_limit_seconds()
    last_activity = time.monotonic()
    try:
        while True:
            try:
                command = commands.get(timeout=1.0)
            except queue.Empty:
                if time.monotonic() - last_activity >= limit:
                    print("ENGINE_IDLE_EXIT=1", flush=True)
                    return 0
                continue
            if command is None or command.get("kind") == "shutdown":
                return 0
            job_id = str(command.get("id", ""))
            kind = str(command.get("kind", ""))
            argv = [str(item) for item in command.get("argv", [])]
            # The worker modules read their correlation id from the environment;
            # this process serves many jobs, so each one sets its own.
            operation_id = str(command.get("operation_id") or "")
            if operation_id:
                os.environ["PATCHLAB_OPERATION_ID"] = operation_id
            try:
                code = engine.dispatch(kind, argv)
            except BaseException as exc:  # a job must never take the engine down silently
                print(_failure_line(kind, exc), flush=True)
                engine.cache.discard()
                code = 1
            last_activity = time.monotonic()
            print(JOB_DONE_PREFIX + json.dumps({"id": job_id, "exit_code": code}), flush=True)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

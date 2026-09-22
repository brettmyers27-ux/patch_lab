#!/usr/bin/env python3
"""Run the consented local-first preset pipeline, then storage-only relay dedup."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.worker_runtime import emit
from core.local_library import default_local_paths, process_linked_folder, relay_from_environment


def main() -> int:
    defaults = default_local_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--db", type=Path, default=defaults["db"])
    parser.add_argument("--audio-root", type=Path, default=defaults["audio"])
    parser.add_argument("--state-dir", type=Path, default=defaults["states"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    from core.diagnostic_env import capture_environment, write_environment
    from core.diagnostics import inherited_operation_id, new_operation_id, recorder
    from core.operation_state import capture_postmortem, start_operation, write_postmortem
    from core.operation_state import LIBRARY_PHASES

    operation_id = inherited_operation_id() or new_operation_id("library")
    tracker = start_operation(
        "render-sound-library",
        subsystem="local-library",
        phases=LIBRARY_PHASES,
        operation_id=operation_id,
        linked_folder=str(args.root),
        render_processes=args.workers,
    )
    try:
        write_environment(
            capture_environment(
                operation="render-sound-library",
                operation_id=operation_id,
                db_path=args.db,
                linked_folder=args.root,
                settings={"render_processes": args.workers},
            )
        )
    except Exception:
        pass

    def _progress(detail: dict) -> None:
        tracker.mark_progress(
            text=str(detail.get("text", "")),
            stage=detail.get("stage"),
            current=detail.get("current"),
            total=detail.get("total"),
        )
        emit("LOCAL_LIBRARY_PROGRESS=" + json.dumps(detail, sort_keys=True))

    try:
        result = process_linked_folder(
            args.root,
            db_path=args.db,
            audio_root=args.audio_root,
            state_dir=args.state_dir,
            relay=relay_from_environment(),
            render_processes=args.workers,
            operation_id=operation_id,
            log=lambda message: print(message, flush=True),
            progress=_progress,
        )
    except BaseException as exc:
        write_postmortem(
            capture_postmortem(tracker, exception=exc, trigger="failure")
        )
        tracker.fail(exc)
        user_message = getattr(exc, "user_message", "")
        if user_message:
            print(f"LOCAL_LIBRARY_USER_ERROR={user_message}", flush=True)
        try:
            recorder().flush(timeout=2.0)
            recorder().close()
        except Exception:
            pass
        raise
    tracker.complete("linked-folder processing finished")
    print("LOCAL_LIBRARY_SUMMARY=" + json.dumps(asdict(result), sort_keys=True), flush=True)
    try:
        recorder().flush(timeout=1.0)
        recorder().close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

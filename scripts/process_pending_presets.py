#!/usr/bin/env python3
"""Process presets already catalogued as pending for one Serum generation.

This exists so that installing a missing Serum later does not cost another full
filesystem walk and re-hash of a multi-thousand-file library. PatchLab already
knows these presets exist; this worker only needs their recorded paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.local_library import default_local_paths, process_pending_for_generation


def main() -> int:
    defaults = default_local_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", choices=("serum1", "serum2"), required=True)
    parser.add_argument("--db", type=Path, default=defaults["db"])
    parser.add_argument("--audio-root", type=Path, default=defaults["audio"])
    parser.add_argument("--state-dir", type=Path, default=defaults["states"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from core.diagnostic_env import capture_environment, write_environment
    from core.diagnostics import inherited_operation_id, new_operation_id, recorder
    from core.operation_state import (
        LIBRARY_PHASES,
        capture_postmortem,
        start_operation,
        write_postmortem,
    )

    operation_id = inherited_operation_id() or new_operation_id("pending")
    tracker = start_operation(
        "process-pending-presets",
        subsystem="local-library",
        phases=LIBRARY_PHASES,
        operation_id=operation_id,
        generation=args.generation,
    )
    try:
        write_environment(
            capture_environment(
                operation="process-pending-presets",
                operation_id=operation_id,
                db_path=args.db,
                target_synth=args.generation,
            )
        )
    except Exception:
        pass

    def progress(detail: dict) -> None:
        tracker.mark_progress(
            text=str(detail.get("text", "")),
            stage=detail.get("stage"),
            current=detail.get("current"),
            total=detail.get("total"),
        )
        print(
            "LOCAL_LIBRARY_PROGRESS=" + json.dumps(detail, sort_keys=True), flush=True
        )

    try:
        result = process_pending_for_generation(
            args.generation,
            db_path=args.db,
            audio_root=args.audio_root,
            state_dir=args.state_dir,
            render_processes=args.workers,
            operation_id=operation_id,
            limit=args.limit,
            log=lambda message: print(message, flush=True),
            progress=progress,
        )
    except BaseException as exc:
        write_postmortem(capture_postmortem(tracker, exception=exc, trigger="failure"))
        tracker.fail(exc)
        user_message = getattr(exc, "user_message", "") or str(exc)
        print(f"LOCAL_LIBRARY_USER_ERROR={user_message}", flush=True)
        try:
            recorder().flush(timeout=2.0)
            recorder().close()
        except Exception:
            pass
        raise
    tracker.complete("pending preset processing finished")
    print(
        "LOCAL_LIBRARY_SUMMARY=" + json.dumps(asdict(result), sort_keys=True), flush=True
    )
    try:
        recorder().flush(timeout=1.0)
        recorder().close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

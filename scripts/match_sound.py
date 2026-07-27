#!/usr/bin/env python3
"""QProcess entry point for one user-initiated Match a Sound request."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.match_workflow import (
    DEFAULT_SESSION_ROOT as DEFAULT_MATCH_SESSION_ROOT,
    run_match_file,
)
from core.factory_match import (
    DEFAULT_SESSION_ROOT as DEFAULT_FACTORY_SESSION_ROOT,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--target-synth", choices=("serum1", "serum2"), default="serum2")
    parser.add_argument("--budget", choices=("quick", "balanced", "best"), default="balanced")
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--factory-only", action="store_true")
    parser.add_argument("--factory-mapping", type=Path)
    parser.add_argument("--local-db", type=Path)
    parser.add_argument("--session-root", type=Path)
    args = parser.parse_args()

    def progress(detail: dict) -> None:
        print("MATCH_PROGRESS=" + json.dumps(detail, separators=(",", ":")), flush=True)

    try:
        if args.factory_only:
            from core.factory_match import run_factory_match_file

            result = run_factory_match_file(
                args.audio,
                target_synth=args.target_synth,
                start_offset_s=args.offset,
                mapping_path=args.factory_mapping,
                local_db_path=args.local_db,
                session_root=args.session_root or DEFAULT_FACTORY_SESSION_ROOT,
                progress_callback=progress,
            )
        else:
            result = run_match_file(
                args.audio,
                target_synth=args.target_synth,
                budget=args.budget,
                start_offset_s=args.offset,
                session_root=args.session_root or DEFAULT_MATCH_SESSION_ROOT,
                progress_callback=progress,
            )
    except Exception as exc:
        print(f"MATCH_ERROR={type(exc).__name__}: {exc}", flush=True)
        return 1
    print("MATCH_RESULT=" + str(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

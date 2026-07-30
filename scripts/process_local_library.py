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
    result = process_linked_folder(
        args.root,
        db_path=args.db,
        audio_root=args.audio_root,
        state_dir=args.state_dir,
        relay=relay_from_environment(),
        render_processes=args.workers,
        log=lambda message: print(message, flush=True),
        progress=lambda detail: print(
            "LOCAL_LIBRARY_PROGRESS=" + json.dumps(detail, sort_keys=True),
            flush=True,
        ),
    )
    print("LOCAL_LIBRARY_SUMMARY=" + json.dumps(asdict(result), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

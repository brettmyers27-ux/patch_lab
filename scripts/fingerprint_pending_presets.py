#!/usr/bin/env python3
"""Fingerprint every fully-rendered preset that has no fingerprint yet.

Catch-up path for audio that was never fingerprinted inline -- e.g. a
standalone render-library run, or a recovered legacy library. Never retrains
any model; only runs the shipped CLAP encoder over audio that already exists.
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

from core.local_library import default_local_paths, fingerprint_pending_presets


def main() -> int:
    defaults = default_local_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=defaults["db"])
    parser.add_argument("--audio-root", type=Path, default=defaults["audio"])
    args = parser.parse_args()
    result = fingerprint_pending_presets(
        args.db,
        args.audio_root,
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

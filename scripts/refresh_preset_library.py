#!/usr/bin/env python3
"""Lightweight Phase 2 source refresh for the Prepare Preset Library screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.library_status import refresh_preset_library
from core.local_library import default_local_paths


def main() -> int:
    defaults = default_local_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--db", type=Path, default=defaults["db"])
    args = parser.parse_args()
    summary = refresh_preset_library(args.root, args.db)
    print("LOCAL_LIBRARY_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

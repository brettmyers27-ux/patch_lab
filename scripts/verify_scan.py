#!/usr/bin/env python3
"""Print the repeatable Milestone 1 database gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.db import DEFAULT_DB_PATH
from core.verify import format_table, library_scan_checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()
    results = library_scan_checks(args.db)
    print("MILESTONE 1 GATE")
    print(format_table(results))
    failed = any(result.failed for result in results)
    print(f"\nGATE: {'FAIL' if failed else 'PASS'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""QProcess entry point for a user-approved PatchLab bug report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.bug_report import load_request
from core.local_library import relay_from_environment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    try:
        request = load_request(args.request)
        relay = relay_from_environment()
        if relay is None:
            raise RuntimeError(
                "PatchLab is not connected to the private support service. "
                "Reconnect and try sending the report again."
            )
        receipt = relay.submit_bug_report(
            ticket_id=request.ticket_id,
            comments=request.comments,
            logs=request.logs,
        )
    except Exception as exc:
        print(f"BUG_REPORT_ERROR={type(exc).__name__}: {exc}", flush=True)
        return 1
    print(
        "BUG_REPORT_RESULT="
        + json.dumps(receipt, separators=(",", ":"), sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

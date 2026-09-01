#!/usr/bin/env python3
"""Check GitHub for a newer published PatchLab version.

Read-only: this never modifies anything. Applying an update is a separate
step (scripts/apply_update.sh), only ever triggered after the user chooses
"Update Now" in the app's dialog.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.__version__ import __version__ as CURRENT_VERSION  # noqa: E402
from core.update_check import fetch_remote_version, update_available  # noqa: E402


def main() -> int:
    remote = fetch_remote_version()
    result = {
        "current_version": CURRENT_VERSION,
        "remote_version": remote,
        "update_available": update_available(CURRENT_VERSION, remote),
    }
    print("UPDATE_CHECK_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

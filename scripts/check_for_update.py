#!/usr/bin/env python3
"""Check whether a real, checksum-verified PatchLab update is available."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.__version__ import __version__ as CURRENT_VERSION  # noqa: E402
from core.update_check import (  # noqa: E402
    fetch_remote_version,
    newest_macos_package,
    update_available,
)


def _private_package_result() -> dict[str, object]:
    """Return an update only when its private PKG exists in the relay catalog."""

    from core.access_gate import stored_relay_credential
    from core.relay_client import RelayClient

    current = CURRENT_VERSION
    url = os.environ.get("PATCHLAB_RELAY_URL", "").strip()
    if os.environ.get("PATCHLAB_DISABLE_RELAY", "").strip() == "1" or not url:
        return {"current_version": current, "remote_version": None, "update_available": False}
    password, token = stored_relay_credential()
    if not password and not token:
        return {"current_version": current, "remote_version": None, "update_available": False}
    try:
        rows = RelayClient(url, password or "", timeout=15.0, token=token).artifact_manifest()
        release = newest_macos_package(rows, current)
    except Exception as exc:
        # Update availability is never allowed to interrupt app launch.  The
        # diagnostic line remains useful in a local support ticket.
        print(f"UPDATE_CHECK_NOTE=Private release catalog unavailable: {type(exc).__name__}: {exc}", flush=True)
        release = None
    if release is None:
        return {"current_version": current, "remote_version": None, "update_available": False}
    return {
        "current_version": current,
        "remote_version": release.version,
        "update_available": True,
        "package": {
            "name": release.name,
            "version": release.version,
            "size": release.size,
            "sha256": release.sha256,
        },
    }


def main() -> int:
    if os.environ.get("PATCHLAB_PACKAGED_INSTALLER") == "1":
        result = _private_package_result()
    else:
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

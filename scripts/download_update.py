#!/usr/bin/env python3
"""Download one authenticated, checksum-pinned PatchLab macOS update."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.access_gate import stored_relay_credential  # noqa: E402
from core.platform_env import ENV  # noqa: E402
from core.relay_client import RelayClient  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    url = os.environ.get("PATCHLAB_RELAY_URL", "").strip()
    password, token = stored_relay_credential()
    if not url or (not password and not token):
        raise SystemExit("Private update access is not available on this device.")
    destination = Path(ENV.app_data_dir) / "updates" / args.name
    last_percent = -1

    def progress(received: int, total: int) -> None:
        nonlocal last_percent
        percent = int(received * 100 / total)
        if percent != last_percent:
            last_percent = percent
            print(
                "UPDATE_DOWNLOAD_PROGRESS="
                + json.dumps({"received": received, "total": total, "percent": percent}),
                flush=True,
            )

    client = RelayClient(url, password or "", timeout=60.0, token=token)
    path = client.download_artifact(
        name=args.name,
        destination=destination,
        size=args.size,
        sha256=args.sha256,
        progress=progress,
    )
    print(
        "UPDATE_DOWNLOAD_RESULT="
        + json.dumps({"path": str(path), "version": args.version}, sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

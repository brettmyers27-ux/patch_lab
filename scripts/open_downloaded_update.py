#!/usr/bin/env python3
"""Open macOS Installer only after the currently-running PatchLab has exited."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pid", required=True, type=int)
    parser.add_argument("--package", required=True, type=Path)
    args = parser.parse_args()
    package = args.package.expanduser().resolve()
    if package.suffix != ".pkg" or not package.is_file():
        raise SystemExit("Downloaded update package is missing.")
    for _ in range(90):
        try:
            os.kill(args.parent_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(1)
    else:
        raise SystemExit("PatchLab did not exit before opening the installer.")
    subprocess.run(["/usr/bin/open", str(package)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

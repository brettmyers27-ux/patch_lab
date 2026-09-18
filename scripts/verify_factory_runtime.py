#!/usr/bin/env python3
"""Verify installed factory preset paths outside PatchLab's UI process."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.factory_verify import verify_local_factory_install
from core.platform_env import ENV


def main() -> int:
    result = verify_local_factory_install(
        mapping_path=ENV.app_data_dir / "factory-paths.json"
    )
    print(
        "FACTORY_VERIFY_RESULT="
        + json.dumps(asdict(result), separators=(",", ":"), sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
